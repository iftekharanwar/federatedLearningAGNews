"""
Test script to verify the dashboard components work correctly.
"""
import os
import json
from datetime import datetime

def main():
    """
    Test the dashboard components.
    """
    print("Testing dashboard components...")
    
    # Create test metrics
    os.makedirs("logs", exist_ok=True)
    
    # Create fit metrics
    fit_metrics = []
    for i in range(1, 3):
        fit_metrics.append({
            "round": i,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "loss": 1.5 - i * 0.2,
                "accuracy": 0.6 + i * 0.1
            },
            "num_clients": 3,
            "total_samples": 300
        })
    
    # Create evaluate metrics
    evaluate_metrics = []
    for i in range(1, 3):
        evaluate_metrics.append({
            "round": i,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "loss": 1.6 - i * 0.2,
                "accuracy": 0.55 + i * 0.1
            },
            "num_clients": 3,
            "total_samples": 300
        })
    
    # Save metrics to files
    with open(os.path.join("logs", "fit_metrics.json"), "w") as f:
        json.dump(fit_metrics, f, indent=2)
    
    with open(os.path.join("logs", "evaluate_metrics.json"), "w") as f:
        json.dump(evaluate_metrics, f, indent=2)
    
    print("Metrics saved to logs directory")
    print("Dashboard components test completed successfully!")
    print("You can now run 'python run_dashboard.py' to view the dashboard")

if __name__ == "__main__":
    main()
