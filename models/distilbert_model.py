"""
DistilBERT model for text classification on AG News dataset.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

class DistilBertForSequenceClassification(nn.Module):
    """
    DistilBERT model with a classification head for text classification.
    """
    def __init__(self, num_labels=4, dropout_rate=0.1, freeze_backbone=False):
        """
        Initialize the model.
        
        Args:
            num_labels: Number of output classes (4 for AG News)
            dropout_rate: Dropout rate for the classification head
            freeze_backbone: Whether to freeze the DistilBERT backbone
        """
        super(DistilBertForSequenceClassification, self).__init__()
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_backbone:
            for param in self.distilbert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        
        nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            logits: Output logits
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

def get_tokenizer():
    """
    Get the DistilBERT tokenizer.
    
    Returns:
        tokenizer: DistilBERT tokenizer
    """
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def get_model(num_labels=4, dropout_rate=0.1, freeze_backbone=False):
    """
    Get the DistilBERT model.
    
    Args:
        num_labels: Number of output classes (4 for AG News)
        dropout_rate: Dropout rate for the classification head
        freeze_backbone: Whether to freeze the DistilBERT backbone
        
    Returns:
        model: DistilBERT model
    """
    return DistilBertForSequenceClassification(
        num_labels=num_labels,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )
