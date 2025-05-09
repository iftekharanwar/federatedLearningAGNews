"""
Dashboard for monitoring federated learning progress on AG News text classification.
"""
import os
import json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime

class FederatedDashboard:
    """
    Dashboard for monitoring federated learning progress.
    """
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def load_metrics(self):
        """
        Load metrics from log files.
        """
        metrics = {
            'fit': [],
            'evaluate': []
        }
        
        for stage in ['fit', 'evaluate']:
            log_file = os.path.join(self.log_dir, f"{stage}_metrics.json")
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    metrics[stage] = json.load(f)
        
        return metrics
    
    def setup_layout(self):
        """
        Set up the dashboard layout.
        """
        self.app.layout = html.Div([
            html.H1("Federated Learning AG News Dashboard"),
            
            html.Div([
                html.H2("Training Progress"),
                dcc.Graph(id='loss-graph'),
                dcc.Graph(id='accuracy-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,  # in milliseconds
                    n_intervals=0
                )
            ]),
            
            html.Div([
                html.H2("Client Participation"),
                dcc.Graph(id='client-participation-graph')
            ]),
            
            html.Div([
                html.H2("Privacy Metrics"),
                dcc.Graph(id='privacy-graph')
            ])
        ])
    
    def setup_callbacks(self):
        """
        Set up the dashboard callbacks.
        """
        @self.app.callback(
            [Output('loss-graph', 'figure'),
             Output('accuracy-graph', 'figure'),
             Output('client-participation-graph', 'figure'),
             Output('privacy-graph', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            metrics = self.load_metrics()
            
            loss_fig = go.Figure()
            
            if metrics['fit']:
                rounds = [m['round'] for m in metrics['fit']]
                if 'loss' in metrics['fit'][0]['metrics']:
                    losses = [m['metrics']['loss'] for m in metrics['fit']]
                    loss_fig.add_trace(go.Scatter(
                        x=rounds, y=losses, mode='lines+markers', name='Training Loss'
                    ))
            
            if metrics['evaluate']:
                rounds = [m['round'] for m in metrics['evaluate']]
                if 'loss' in metrics['evaluate'][0]['metrics']:
                    losses = [m['metrics']['loss'] for m in metrics['evaluate']]
                    loss_fig.add_trace(go.Scatter(
                        x=rounds, y=losses, mode='lines+markers', name='Validation Loss'
                    ))
            
            loss_fig.update_layout(
                title='Loss per Round',
                xaxis_title='Round',
                yaxis_title='Loss',
                legend_title='Legend'
            )
            
            accuracy_fig = go.Figure()
            
            if metrics['fit']:
                rounds = [m['round'] for m in metrics['fit']]
                if 'accuracy' in metrics['fit'][0]['metrics']:
                    accuracies = [m['metrics']['accuracy'] for m in metrics['fit']]
                    accuracy_fig.add_trace(go.Scatter(
                        x=rounds, y=accuracies, mode='lines+markers', name='Training Accuracy'
                    ))
            
            if metrics['evaluate']:
                rounds = [m['round'] for m in metrics['evaluate']]
                if 'accuracy' in metrics['evaluate'][0]['metrics']:
                    accuracies = [m['metrics']['accuracy'] for m in metrics['evaluate']]
                    accuracy_fig.add_trace(go.Scatter(
                        x=rounds, y=accuracies, mode='lines+markers', name='Validation Accuracy'
                    ))
            
            accuracy_fig.update_layout(
                title='Accuracy per Round',
                xaxis_title='Round',
                yaxis_title='Accuracy',
                legend_title='Legend'
            )
            
            client_fig = go.Figure()
            
            if metrics['fit']:
                rounds = [m['round'] for m in metrics['fit']]
                num_clients = [m['num_clients'] for m in metrics['fit']]
                
                client_fig.add_trace(go.Bar(
                    x=rounds, y=num_clients, name='Number of Clients'
                ))
            
            client_fig.update_layout(
                title='Client Participation per Round',
                xaxis_title='Round',
                yaxis_title='Number of Clients',
                legend_title='Legend'
            )
            
            privacy_fig = go.Figure()
            
            if metrics['fit']:
                rounds = [m['round'] for m in metrics['fit']]
                epsilons = [1.0] * len(rounds)  # Placeholder for actual privacy budget tracking
                
                privacy_fig.add_trace(go.Scatter(
                    x=rounds, y=epsilons, mode='lines+markers', name='Privacy Budget (Epsilon)'
                ))
            
            privacy_fig.update_layout(
                title='Privacy Budget per Round',
                xaxis_title='Round',
                yaxis_title='Epsilon',
                legend_title='Legend'
            )
            
            return loss_fig, accuracy_fig, client_fig, privacy_fig
    
    def run_server(self, debug=True, port=8050):
        """
        Run the dashboard server.
        """
        self.app.run(debug=debug, port=port)


if __name__ == '__main__':
    dashboard = FederatedDashboard()
    dashboard.run_server()
