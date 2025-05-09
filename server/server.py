"""
Federated learning server for AG News text classification.
"""
import os
import json
from datetime import datetime
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union

class AGNewsFederatedServer:
    """
    Federated learning server for AG News text classification.
    """
    def __init__(
        self,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3,
        num_rounds=5,
        fraction_fit=1.0,
        fraction_eval=1.0,
        log_dir="logs"
    ):
        """
        Initialize the server.
        
        Args:
            min_fit_clients: Minimum number of clients for training
            min_eval_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of available clients
            num_rounds: Number of federated learning rounds
            fraction_fit: Fraction of clients to use for training
            fraction_eval: Fraction of clients to use for evaluation
            log_dir: Directory to save logs
        """
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.log_dir = log_dir
        self.current_round = 0  # Track current round
        self.fit_round = 0      # Track fit round
        self.eval_round = 0     # Track eval round
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.strategy = FedAvg(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_eval,
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn
        )
    
    def fit_config(self, server_round: int):
        """
        Return training configuration for clients.
        
        Args:
            server_round: Current round of federated learning
            
        Returns:
            config: Configuration dictionary
        """
        return {
            "server_round": server_round,
            "local_epochs": 1
        }
    
    def evaluate_config(self, server_round: int):
        """
        Return evaluation configuration for clients.
        
        Args:
            server_round: Current round of federated learning
            
        Returns:
            config: Configuration dictionary
        """
        return {
            "server_round": server_round
        }
    
    def fit_metrics_aggregation_fn(
        self,
        fit_metrics: List[Tuple[int, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Aggregate training metrics from clients.
        
        Args:
            fit_metrics: List of tuples (num_samples, metrics_dict)
            
        Returns:
            aggregated_metrics: Aggregated metrics dictionary
        """
        self.fit_round += 1
        
        num_samples_list = [num_samples for num_samples, _ in fit_metrics]
        loss_list = [metrics["loss"] * num_samples for num_samples, metrics in fit_metrics]
        accuracy_list = [metrics["accuracy"] * num_samples for num_samples, metrics in fit_metrics]
        
        total_samples = sum(num_samples_list)
        aggregated_loss = sum(loss_list) / total_samples if total_samples > 0 else 0.0
        aggregated_accuracy = sum(accuracy_list) / total_samples if total_samples > 0 else 0.0
        
        aggregated_metrics = {
            "loss": aggregated_loss,
            "accuracy": aggregated_accuracy
        }
        
        try:
            log_entry = {
                "round": self.fit_round,
                "timestamp": datetime.now().isoformat(),
                "metrics": aggregated_metrics,
                "num_clients": len(fit_metrics),
                "total_samples": total_samples
            }
            
            log_file = os.path.join(self.log_dir, "fit_metrics.json")
            logs = []
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            
            logs.append(log_entry)
            
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
            
            print(f"Fit metrics for round {self.fit_round} saved to {log_file}")
        
        except Exception as e:
            print(f"Error saving fit metrics: {e}")
        
        return aggregated_metrics
    
    def evaluate_metrics_aggregation_fn(
        self,
        eval_metrics: List[Tuple[int, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from clients.
        
        Args:
            eval_metrics: List of tuples (num_samples, metrics_dict)
            
        Returns:
            aggregated_metrics: Aggregated metrics dictionary
        """
        self.eval_round += 1
        
        num_samples_list = [num_samples for num_samples, _ in eval_metrics]
        loss_list = [metrics["loss"] * num_samples for num_samples, metrics in eval_metrics]
        accuracy_list = [metrics["accuracy"] * num_samples for num_samples, metrics in eval_metrics]
        
        total_samples = sum(num_samples_list)
        aggregated_loss = sum(loss_list) / total_samples if total_samples > 0 else 0.0
        aggregated_accuracy = sum(accuracy_list) / total_samples if total_samples > 0 else 0.0
        
        aggregated_metrics = {
            "loss": aggregated_loss,
            "accuracy": aggregated_accuracy
        }
        
        try:
            log_entry = {
                "round": self.eval_round,
                "timestamp": datetime.now().isoformat(),
                "metrics": aggregated_metrics,
                "num_clients": len(eval_metrics),
                "total_samples": total_samples
            }
            
            log_file = os.path.join(self.log_dir, "evaluate_metrics.json")
            logs = []
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        logs = json.load(f)
                except json.JSONDecodeError:
                    logs = []
            
            logs.append(log_entry)
            
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
            
            print(f"Evaluate metrics for round {self.eval_round} saved to {log_file}")
        
        except Exception as e:
            print(f"Error saving evaluate metrics: {e}")
        
        return aggregated_metrics
    
    def start_server(self):
        """
        Start the federated learning server.
        """
        fl.server.start_server(
            server_address="0.0.0.0:8081",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy
        )
