"""
Main script to run the federated learning system for AG News text classification.
"""
import os
import argparse
import torch
import flwr as fl
from clients.client import AGNewsClient
from server.server import AGNewsFederatedServer
import multiprocessing
import time
import random
import numpy as np

os.environ["GRPC_MAX_SEND_MESSAGE_LENGTH"] = str(1024 * 1024 * 500)  # 500MB --> alter it according to the size of the data
os.environ["GRPC_MAX_RECEIVE_MESSAGE_LENGTH"] = str(1024 * 1024 * 500)  # 500MB

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def start_client(provider_id, data_dir, batch_size, learning_rate, epochs, 
                 max_length, use_dp, dp_epsilon, sample_size, freeze_backbone):
    """
    Start a federated learning client.
    """
    client = AGNewsClient(
        provider_id=provider_id,
        data_dir=data_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        max_length=max_length,
        use_dp=use_dp,
        dp_epsilon=dp_epsilon,
        sample_size=sample_size,
        freeze_backbone=freeze_backbone
    )
    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=client)

def start_server(num_providers, num_rounds, log_dir):
    """
    Start the federated learning server.
    """
    print(f"Server: Initializing with {num_providers} providers, {num_rounds} rounds...")
    server = AGNewsFederatedServer(
        min_fit_clients=num_providers,
        min_eval_clients=num_providers,
        min_available_clients=num_providers,
        num_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_eval=1.0,
        log_dir=log_dir
    )
    print(f"Server: Starting server...")
    server.start_server()
    print(f"Server: Server completed after {num_rounds} rounds.")

def main():
    """
    Run the federated learning system.
    """
    print("Starting main function...")
    parser = argparse.ArgumentParser(description='Run federated learning for AG News text classification')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of local epochs')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--num_rounds', type=int, default=5,
                        help='Number of federated learning rounds')
    parser.add_argument('--num_providers', type=int, default=3,
                        help='Number of simulated providers')
    parser.add_argument('--use_dp', action='store_true',
                        help='Use differential privacy')
    parser.add_argument('--dp_epsilon', type=float, default=1.0,
                        help='Epsilon value for differential privacy')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use from each provider (memory efficiency)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze the DistilBERT backbone')
    
    print("Parsing arguments...")
    args = parser.parse_args()
    
    print("Creating log directory...")
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Starting federated learning with {args.num_providers} providers, {args.num_rounds} rounds")
    print(f"Using DistilBERT model with max_length={args.max_length}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    if args.sample_size:
        print(f"Memory efficiency settings: sample_size={args.sample_size}")

    print("Creating server process...")
    server_process = multiprocessing.Process(
        target=start_server,
        args=(args.num_providers, args.num_rounds, args.log_dir)
    )
    
    print("Starting server process...")
    server_process.start()
    
    print("Server started, waiting for initialization...")
    time.sleep(3)
    
    client_processes = []
    for i in range(args.num_providers):
        provider_id = f"provider{i+1}"
        print(f"Creating client process for {provider_id}...")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(
                provider_id,
                args.data_dir,
                args.batch_size,
                args.learning_rate,
                args.epochs,
                args.max_length,
                args.use_dp,
                args.dp_epsilon,
                args.sample_size,
                args.freeze_backbone
            )
        )
        client_processes.append(client_process)
        print(f"Starting client process for {provider_id}...")
        client_process.start()
        print(f"Client process for {provider_id} started.")
    
    print(f"All {args.num_providers} clients started, waiting for training to complete...")
    try:
        server_process.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, terminating processes...")
    finally:
        print("Terminating client processes...")
        for process in client_processes:
            process.terminate()
    
    print("Federated learning complete!")

if __name__ == "__main__":
    main()
