"""Asset Tracking System - Streamlit Demo Application.

This demo simulates real-time asset tracking with edge constraints,
showing model performance, metrics, and deployment scenarios.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Asset Tracking System Demo",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class AssetTrackingDemo:
    """Main demo class for asset tracking simulation."""
    
    def __init__(self):
        """Initialize the demo."""
        self.setup_session_state()
        self.load_config()
        self.initialize_models()
    
    def setup_session_state(self):
        """Setup Streamlit session state variables."""
        if 'asset_data' not in st.session_state:
            st.session_state.asset_data = []
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'metrics_history' not in st.session_state:
            st.session_state.metrics_history = []
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
    
    def load_config(self):
        """Load configuration."""
        try:
            with open('configs/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Default config if file not found
            self.config = {
                'model': {'input_features': 4},
                'data': {'synthetic_data': {}},
                'demo': {'simulation': {'update_interval': 1.0}}
            }
    
    def initialize_models(self):
        """Initialize demo models."""
        # Simple mock models for demo
        self.baseline_model = self.create_mock_model("baseline")
        self.edge_model = self.create_mock_model("edge")
    
    def create_mock_model(self, model_type: str):
        """Create a mock model for demo purposes."""
        class MockModel:
            def __init__(self, model_type: str):
                self.model_type = model_type
                self.accuracy = 0.95 if model_type == "baseline" else 0.92
                self.latency = 0.05 if model_type == "baseline" else 0.02
                self.model_size = 2.5 if model_type == "baseline" else 0.8
            
            def predict(self, features: np.ndarray) -> Tuple[int, float]:
                """Mock prediction."""
                # Simple rule-based prediction for demo
                location_var, signal_str, days_inactive, motion_count = features
                
                if location_var < 3 and signal_str < 60 and days_inactive > 10:
                    prediction = 1  # Lost
                    confidence = 0.9
                else:
                    prediction = 0  # Normal
                    confidence = 0.8
                
                # Add some randomness
                confidence += np.random.normal(0, 0.1)
                confidence = np.clip(confidence, 0.1, 0.99)
                
                return prediction, confidence
        
        return MockModel(model_type)
    
    def generate_asset_data(self) -> Tuple[np.ndarray, int]:
        """Generate synthetic asset data."""
        # Generate features
        location_variance = np.random.normal(5, 2)
        signal_strength = np.random.normal(70, 10)
        days_inactive = np.random.randint(0, 30)
        motion_count = np.random.poisson(3)
        
        features = np.array([location_variance, signal_strength, days_inactive, motion_count])
        
        # Generate true label
        true_label = 1 if (location_variance < 3 and signal_strength < 60 and days_inactive > 10) else 0
        
        return features, true_label
    
    def run_simulation(self, n_samples: int = 100):
        """Run asset tracking simulation."""
        st.session_state.simulation_running = True
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(n_samples):
            # Generate data
            features, true_label = self.generate_asset_data()
            
            # Get predictions from both models
            baseline_pred, baseline_conf = self.baseline_model.predict(features)
            edge_pred, edge_conf = self.edge_model.predict(features)
            
            # Store data
            timestamp = time.time()
            asset_data = {
                'timestamp': timestamp,
                'location_variance': features[0],
                'signal_strength': features[1],
                'days_inactive': features[2],
                'motion_count': features[3],
                'true_label': true_label,
                'baseline_prediction': baseline_pred,
                'baseline_confidence': baseline_conf,
                'edge_prediction': edge_pred,
                'edge_confidence': edge_conf,
            }
            
            st.session_state.asset_data.append(asset_data)
            
            # Update progress
            progress_bar.progress((i + 1) / n_samples)
            status_text.text(f"Processing asset {i + 1}/{n_samples}")
            
            # Small delay for realistic simulation
            time.sleep(0.1)
        
        st.session_state.simulation_running = False
        status_text.text("Simulation completed!")
    
    def display_metrics_dashboard(self):
        """Display metrics dashboard."""
        if not st.session_state.asset_data:
            st.warning("No data available. Run simulation first.")
            return
        
        df = pd.DataFrame(st.session_state.asset_data)
        
        # Calculate metrics
        baseline_acc = (df['baseline_prediction'] == df['true_label']).mean()
        edge_acc = (df['edge_prediction'] == df['true_label']).mean()
        
        baseline_avg_conf = df['baseline_confidence'].mean()
        edge_avg_conf = df['edge_confidence'].mean()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Baseline Accuracy", f"{baseline_acc:.3f}")
        with col2:
            st.metric("Edge Accuracy", f"{edge_acc:.3f}")
        with col3:
            st.metric("Baseline Avg Confidence", f"{baseline_avg_conf:.3f}")
        with col4:
            st.metric("Edge Avg Confidence", f"{edge_avg_conf:.3f}")
        
        # Performance comparison
        st.subheader("Model Performance Comparison")
        
        performance_data = {
            'Model': ['Baseline', 'Edge'],
            'Accuracy': [baseline_acc, edge_acc],
            'Avg Confidence': [baseline_avg_conf, edge_avg_conf],
            'Latency (ms)': [self.baseline_model.latency * 1000, self.edge_model.latency * 1000],
            'Model Size (MB)': [self.baseline_model.model_size, self.edge_model.model_size],
        }
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
    
    def display_visualizations(self):
        """Display data visualizations."""
        if not st.session_state.asset_data:
            st.warning("No data available. Run simulation first.")
            return
        
        df = pd.DataFrame(st.session_state.asset_data)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Location Variance', 'Signal Strength', 'Days Inactive', 'Motion Count'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        features = ['location_variance', 'signal_strength', 'days_inactive', 'motion_count']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for feature, (row, col) in zip(features, positions):
            fig.add_trace(
                go.Histogram(x=df[feature], name=feature.replace('_', ' ').title()),
                row=row, col=col
            )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction accuracy over time
        st.subheader("Prediction Accuracy Over Time")
        
        df['baseline_correct'] = (df['baseline_prediction'] == df['true_label']).astype(int)
        df['edge_correct'] = (df['edge_prediction'] == df['true_label']).astype(int)
        
        # Rolling accuracy
        window_size = min(20, len(df) // 4)
        df['baseline_rolling_acc'] = df['baseline_correct'].rolling(window=window_size).mean()
        df['edge_rolling_acc'] = df['edge_correct'].rolling(window=window_size).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df['baseline_rolling_acc'],
            mode='lines', name='Baseline Model',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['edge_rolling_acc'],
            mode='lines', name='Edge Model',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Rolling Accuracy Comparison",
            xaxis_title="Sample Index",
            yaxis_title="Accuracy",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_edge_constraints(self):
        """Display edge deployment constraints."""
        st.subheader("Edge Deployment Constraints")
        
        # Device configurations
        devices = {
            'Raspberry Pi 4': {'cpu_cores': 4, 'memory_gb': 4, 'power_w': 3.4},
            'Jetson Nano': {'cpu_cores': 4, 'memory_gb': 4, 'power_w': 5.0},
            'Android Mobile': {'cpu_cores': 8, 'memory_gb': 6, 'power_w': 2.0},
            'MCU Device': {'cpu_cores': 2, 'memory_mb': 4, 'power_mw': 100},
        }
        
        device_df = pd.DataFrame(devices).T
        st.dataframe(device_df, use_container_width=True)
        
        # Model size comparison
        st.subheader("Model Size Comparison")
        
        model_sizes = {
            'Model': ['Baseline', 'Edge Optimized'],
            'Size (MB)': [self.baseline_model.model_size, self.edge_model.model_size],
            'Parameters': ['~50K', '~15K'],
            'Latency (ms)': [self.baseline_model.latency * 1000, self.edge_model.latency * 1000],
        }
        
        size_df = pd.DataFrame(model_sizes)
        st.dataframe(size_df, use_container_width=True)


def main():
    """Main demo application."""
    # Header
    st.markdown('<h1 class="main-header">Asset Tracking System Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ DISCLAIMER:</strong> This is a research and educational demonstration. 
    This system is NOT intended for safety-critical applications or production deployment 
    without proper validation and testing.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize demo
    demo = AssetTrackingDemo()
    
    # Sidebar controls
    st.sidebar.header("Simulation Controls")
    
    n_samples = st.sidebar.slider("Number of Samples", 10, 500, 100)
    
    if st.sidebar.button("Run Simulation", disabled=st.session_state.simulation_running):
        demo.run_simulation(n_samples)
    
    if st.sidebar.button("Clear Data"):
        st.session_state.asset_data = []
        st.session_state.predictions = []
        st.session_state.metrics_history = []
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Visualizations", "Edge Constraints", "About"])
    
    with tab1:
        demo.display_metrics_dashboard()
    
    with tab2:
        demo.display_visualizations()
    
    with tab3:
        demo.display_edge_constraints()
    
    with tab4:
        st.markdown("""
        ## About This Demo
        
        This demo showcases an Edge AI Asset Tracking System that monitors the location 
        and status of valuable assets using IoT sensors.
        
        ### Features:
        - **Real-time Simulation**: Generates synthetic sensor data (GPS, BLE, accelerometer)
        - **Dual Model Comparison**: Baseline vs Edge-optimized neural networks
        - **Performance Metrics**: Accuracy, latency, model size, and efficiency
        - **Edge Constraints**: Device-specific deployment considerations
        - **Visual Analytics**: Interactive charts and performance monitoring
        
        ### Use Cases:
        - Logistics and fleet management
        - Warehouse asset tracking
        - Equipment monitoring
        - Supply chain visibility
        
        ### Technical Stack:
        - **Models**: PyTorch neural networks with compression techniques
        - **Deployment**: ONNX, TensorFlow Lite, CoreML for edge devices
        - **Visualization**: Streamlit with Plotly charts
        - **Simulation**: Synthetic sensor data generation
        
        ### Model Types:
        1. **Baseline Model**: Full-precision neural network (higher accuracy)
        2. **Edge Model**: Optimized for edge deployment (lower latency, smaller size)
        
        This system demonstrates the trade-offs between accuracy and efficiency 
        in edge AI applications.
        """)


if __name__ == "__main__":
    main()
