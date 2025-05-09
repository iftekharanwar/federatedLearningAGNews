"""
Run the federated learning dashboard for AG News text classification.
"""
import argparse
from dashboard.dashboard import FederatedDashboard

def main():
    """
    Run the federated learning dashboard.
    """
    parser = argparse.ArgumentParser(description='Run the federated learning dashboard')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing log files')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    print(f"Starting dashboard with logs from {args.log_dir} on port {args.port}...")
    print(f"Debug mode: {args.debug}")
    
    dashboard = FederatedDashboard(log_dir=args.log_dir)
    dashboard.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()
