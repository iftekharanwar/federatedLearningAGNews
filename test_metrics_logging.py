"""
Test script to verify metrics logging fixes.
"""
import os
import json
from datetime import datetime
from server.server import AGNewsFederatedServer

def test_metrics_logging():
    """
    Test metrics logging functionality.
    """
    test_log_dir = "test_logs"
    os.makedirs(test_log_dir, exist_ok=True)
    
    server = AGNewsFederatedServer(
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3,
        num_rounds=2,
        log_dir=test_log_dir
    )
    
    fit_metrics = [
        (100, {"loss": 1.5, "accuracy": 0.6}),
        (150, {"loss": 1.3, "accuracy": 0.7}),
        (120, {"loss": 1.4, "accuracy": 0.65})
    ]
    
    aggregated_fit = server.fit_metrics_aggregation_fn(fit_metrics)
    print(f"Aggregated fit metrics: {aggregated_fit}")
    
    eval_metrics = [
        (80, {"loss": 1.4, "accuracy": 0.65}),
        (90, {"loss": 1.2, "accuracy": 0.75}),
        (85, {"loss": 1.3, "accuracy": 0.7})
    ]
    
    aggregated_eval = server.evaluate_metrics_aggregation_fn(eval_metrics)
    print(f"Aggregated evaluate metrics: {aggregated_eval}")
    
    fit_log_file = os.path.join(test_log_dir, "fit_metrics.json")
    eval_log_file = os.path.join(test_log_dir, "evaluate_metrics.json")
    
    if os.path.exists(fit_log_file):
        with open(fit_log_file, "r") as f:
            fit_logs = json.load(f)
            print(f"Fit logs: {fit_logs}")
    else:
        print(f"Error: Fit log file not created at {fit_log_file}")
    
    if os.path.exists(eval_log_file):
        with open(eval_log_file, "r") as f:
            eval_logs = json.load(f)
            print(f"Evaluate logs: {eval_logs}")
    else:
        print(f"Error: Evaluate log file not created at {eval_log_file}")
    
    if os.path.exists(fit_log_file):
        os.remove(fit_log_file)
    if os.path.exists(eval_log_file):
        os.remove(eval_log_file)
    os.rmdir(test_log_dir)

if __name__ == "__main__":
    test_metrics_logging()
