"""
Test script to verify the federated learning components work correctly.
"""
import os
import torch
import numpy as np
from collections import OrderedDict
from transformers import DistilBertTokenizer
from models.distilbert_model import get_model
from utils.data_utils import load_data

def main():
    """
    Test the federated learning components.
    """
    print("Testing federated learning components...")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = get_model(num_labels=4, freeze_backbone=True)
    
    # Test data loading
    provider_dir = os.path.join("data/processed", "provider1")
    
    print(f"Loading data from {provider_dir}...")
    
    # Use a small sample size for testing
    train_loader = load_data(
        os.path.join(provider_dir, "train.csv"),
        tokenizer,
        batch_size=4,
        max_length=128,
        sample_size=100
    )
    
    print(f"Loaded {len(train_loader.dataset)} training samples")
    
    # Test model parameter extraction
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    print(f"Extracted {len(params)} model parameters")
    
    # Test model parameter setting
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    print("Successfully set model parameters")
    
    # Test forward pass with batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['label']
    
    print(f"Batch shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}, labels={labels.shape}")
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Output shape: {outputs.shape}")
    
    # Test loss calculation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    print(f"Loss: {loss.item()}")
    
    print("Federated learning components test completed successfully!")

if __name__ == "__main__":
    main()
