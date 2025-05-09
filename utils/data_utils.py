"""
Utility functions for data processing in the AG News federated learning project.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer

class AGNewsDataset(Dataset):
    """
    Dataset class for AG News.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Get the number of samples.
        
        Returns:
            length: Number of samples
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            sample: Dictionary containing input_ids, attention_mask, and label
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path, tokenizer, batch_size=16, max_length=128, sample_size=None):
    """
    Load and prepare data for training or evaluation.
    
    Args:
        data_path: Path to the CSV file
        tokenizer: DistilBERT tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        sample_size: Number of samples to use (for memory efficiency)
        
    Returns:
        dataloader: DataLoader for the dataset
    """
    df = pd.read_csv(data_path)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    dataset = AGNewsDataset(texts, labels, tokenizer, max_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return dataloader

def get_class_names():
    """
    Get the class names for AG News.
    
    Returns:
        class_names: List of class names
    """
    return ["World", "Sports", "Business", "Sci/Tech"]
