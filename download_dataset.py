"""
Download and prepare the AG News dataset for federated learning.
"""
import os
import argparse
import random
import numpy as np
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def download_agnews(output_dir="data/raw"):
    """
    Download the AG News dataset using the Hugging Face datasets library.
    
    Args:
        output_dir: Directory to save the dataset
    """
    print(f"Downloading AG News dataset to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset("ag_news")
    
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    
    train_df.to_csv(os.path.join(output_dir, "agnews_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "agnews_test.csv"), index=False)
    
    print(f"Downloaded AG News dataset: {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"Dataset saved to {output_dir}")
    
    return train_df, test_df

def partition_data(train_df, test_df, num_providers=3, output_dir="data/processed"):
    """
    Partition the AG News dataset across multiple simulated providers.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        num_providers: Number of simulated providers
        output_dir: Directory to save the partitioned data
    """
    print(f"Partitioning data across {num_providers} providers...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_counts = train_df["label"].value_counts()
    print(f"Class distribution: {class_counts.to_dict()}")
    
    
    provider_data = {}
    
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    chunk_size = len(train_df) // num_providers
    
    for i in range(num_providers):
        provider_name = f"provider{i+1}"
        provider_dir = os.path.join(output_dir, provider_name)
        os.makedirs(provider_dir, exist_ok=True)
        
        if i < num_providers - 1:
            provider_train = train_df.iloc[i*chunk_size:(i+1)*chunk_size].copy()
        else:
            provider_train = train_df.iloc[i*chunk_size:].copy()
        
        p_train, p_val = train_test_split(provider_train, test_size=0.2, random_state=RANDOM_SEED)
        
        p_train.to_csv(os.path.join(provider_dir, "train.csv"), index=False)
        p_val.to_csv(os.path.join(provider_dir, "val.csv"), index=False)
        
        test_df.to_csv(os.path.join(provider_dir, "test.csv"), index=False)
        
        print(f"Provider {provider_name}: {len(p_train)} training samples, {len(p_val)} validation samples")
        
        provider_data[provider_name] = {
            "train": p_train,
            "val": p_val,
            "test": test_df
        }
    
    test_df.to_csv(os.path.join(output_dir, "global_test.csv"), index=False)
    
    print(f"Data partitioning complete. Files saved to {output_dir}")
    
    return provider_data

def main():
    """
    Main function to download and partition the AG News dataset.
    """
    parser = argparse.ArgumentParser(description="Download and partition the AG News dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw",
                        help="Directory to save the raw dataset")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Directory to save the processed dataset")
    parser.add_argument("--num_providers", type=int, default=3,
                        help="Number of simulated providers")
    
    args = parser.parse_args()
    
    train_df, test_df = download_agnews(args.raw_dir)
    
    partition_data(train_df, test_df, args.num_providers, args.processed_dir)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
