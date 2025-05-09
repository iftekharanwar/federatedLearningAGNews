"""
Federated learning client for AG News text classification.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import flwr as fl
from transformers import DistilBertTokenizer
from opacus import PrivacyEngine

from models.distilbert_model import get_model
from utils.data_utils import load_data

class AGNewsClient(fl.client.NumPyClient):
    """
    Federated learning client for AG News text classification.
    """
    def __init__(
        self,
        provider_id,
        data_dir,
        batch_size=16,
        learning_rate=5e-5,
        epochs=1,
        max_length=128,
        use_dp=False,
        dp_epsilon=1.0,
        sample_size=None,
        freeze_backbone=True
    ):
        """
        Initialize the client.
        
        Args:
            provider_id: ID of the provider
            data_dir: Directory containing processed data
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            epochs: Number of local epochs
            max_length: Maximum sequence length
            use_dp: Whether to use differential privacy
            dp_epsilon: Epsilon value for differential privacy
            sample_size: Number of samples to use (for memory efficiency)
            freeze_backbone: Whether to freeze the DistilBERT backbone
        """
        self.provider_id = provider_id
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.sample_size = sample_size
        self.freeze_backbone = freeze_backbone
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        self.model = get_model(num_labels=4, freeze_backbone=self.freeze_backbone)
        self.model.to(self.device)
        
        provider_dir = os.path.join(data_dir, provider_id)
        self.train_loader = load_data(
            os.path.join(provider_dir, "train.csv"),
            self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            sample_size=sample_size
        )
        self.val_loader = load_data(
            os.path.join(provider_dir, "val.csv"),
            self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            sample_size=sample_size
        )
        self.test_loader = load_data(
            os.path.join(provider_dir, "test.csv"),
            self.tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            sample_size=sample_size
        )
        
        print(f"Client {provider_id} initialized with {len(self.train_loader.dataset)} training samples")
    
    def get_parameters(self, config):
        """
        Get model parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            parameters: Model parameters as a list of NumPy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters.
        
        Args:
            parameters: Model parameters as a list of NumPy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model on the local dataset.
        
        Args:
            parameters: Model parameters as a list of NumPy arrays
            config: Configuration dictionary
            
        Returns:
            parameters: Updated model parameters
            num_samples: Number of training samples
            metrics: Training metrics
        """
        self.set_parameters(parameters)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        if self.use_dp:
            privacy_engine = PrivacyEngine()
            self.model, optimizer, self.train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                noise_multiplier=1.1,
                max_grad_norm=1.0
            )
            print(f"Using differential privacy with epsilon={self.dp_epsilon}")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_total += labels.size(0)
            
            epoch_loss = epoch_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total
            
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            correct += epoch_correct
            total += epoch_total
        
        avg_loss = total_loss / self.epochs
        accuracy = correct / total
        
        return self.get_parameters(config), len(self.train_loader.dataset), {
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        }
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on the local test dataset.
        
        Args:
            parameters: Model parameters as a list of NumPy arrays
            config: Configuration dictionary
            
        Returns:
            loss: Evaluation loss
            num_samples: Number of evaluation samples
            metrics: Evaluation metrics
        """
        self.set_parameters(parameters)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return float(avg_loss), total, {
            "loss": float(avg_loss),
            "accuracy": float(accuracy)
        }
