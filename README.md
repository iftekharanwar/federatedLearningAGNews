# Federated Learning for AG News Text Classification

This project implements a federated learning system for text classification on the AG News dataset using the Flower framework and DistilBERT model.

## Overview

The system simulates multiple providers (clients) that train local DistilBERT models on their partitions of the AG News dataset. The models are then aggregated by a central server to create a global model without sharing raw data.

## Features

- **Federated Learning**: Train models across multiple simulated providers without sharing raw data
- **DistilBERT Model**: Efficient transformer model for text classification
- **Differential Privacy**: Optional privacy protection using Opacus
- **Dashboard**: Real-time monitoring of training progress
- **Memory Efficiency**: Options to control memory usage for large datasets

## Project Structure

```
fl_agnews_mvp/
├── data/                  # Data directory
│   ├── raw/               # Raw AG News dataset
│   └── processed/         # Processed and partitioned data
├── models/                # Model definitions
│   └── distilbert_model.py # DistilBERT model implementation
├── clients/               # Client implementations
│   └── client.py          # Federated learning client
├── server/                # Server implementations
│   └── server.py          # Federated learning server
├── utils/                 # Utility functions
│   └── data_utils.py      # Data processing utilities
├── dashboard/             # Dashboard for monitoring
│   └── dashboard.py       # Dashboard implementation
├── download_dataset.py    # Script to download and partition the dataset
├── main.py                # Main script to run the federated learning system
├── run_dashboard.py       # Script to run the dashboard
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Download and Partition the Dataset

```bash
python download_dataset.py --raw_dir data/raw --processed_dir data/processed --num_providers 3
```

### 2. Run the Federated Learning System

```bash
python main.py --data_dir data/processed --num_providers 3 --num_rounds 5 --freeze_backbone
```

### 3. Run the Dashboard

```bash
python run_dashboard.py --log_dir logs --port 8050
```

## Configuration Options

- `--num_providers`: Number of simulated providers (default: 3)
- `--num_rounds`: Number of federated learning rounds (default: 5)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate for training (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 128)
- `--use_dp`: Use differential privacy (flag)
- `--dp_epsilon`: Epsilon value for differential privacy (default: 1.0)
- `--sample_size`: Number of samples to use from each provider (memory efficiency)
- `--freeze_backbone`: Freeze the DistilBERT backbone (flag)

## Dataset

The AG News dataset consists of news articles categorized into 4 classes:
1. World
2. Sports
3. Business
4. Sci/Tech

The dataset contains 120,000 training examples and 7,600 test examples.

## Model

The system uses DistilBERT, a distilled version of BERT that is smaller, faster, and retains 97% of BERT's language understanding capabilities. The model is fine-tuned for text classification with a classification head on top.

## Federated Learning Workflow

1. Each provider trains a local DistilBERT model on its partition of the AG News dataset
2. The server aggregates the model updates using FedAvg
3. The updated global model is sent back to the providers
4. The process repeats for multiple rounds
5. The final global model can classify news articles without having seen the raw data

## Privacy and Security

- Raw data never leaves the providers
- Optional differential privacy using Opacus
- Secure aggregation through the Flower framework

## Dashboard

The dashboard provides real-time monitoring of:
- Training and validation loss
- Training and validation accuracy
- Client participation
- Privacy budget (epsilon)

## References

- [Flower Framework](https://flower.dev/)
- [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
- [Opacus](https://opacus.ai/)

