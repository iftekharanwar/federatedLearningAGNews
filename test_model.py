"""
Test script to verify the DistilBERT model works correctly.
"""
import torch
from transformers import DistilBertTokenizer
from models.distilbert_model import get_model

def main():
    """
    Test the DistilBERT model.
    """
    print("Testing DistilBERT model...")
    
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = get_model(num_labels=4, freeze_backbone=True)
    
    # Test input
    text = "This is a test news article about technology and science."
    
    # Tokenize input
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Print results
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output: {outputs}")
    
    print("Model test completed successfully!")

if __name__ == "__main__":
    main()
