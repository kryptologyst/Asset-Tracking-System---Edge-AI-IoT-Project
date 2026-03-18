#!/usr/bin/env python3
"""Asset Tracking System - Deployment Script.

This script demonstrates model deployment to different edge devices
and runtime environments.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models import create_model
from export import create_exporter, create_deployment_manager
from utils import setup_logging, get_device, Timer


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Asset Tracking Models")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--model-type", default="edge", choices=["baseline", "edge"], help="Model type")
    parser.add_argument("--output-dir", default="assets/deployed", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(args.model_type)
    model.eval()
    
    # Create exporter
    exporter = create_exporter(device)
    
    # Export models
    logger.info("Exporting models to edge formats...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with Timer("Model Export"):
        exported_paths = exporter.export_all_formats(
            model,
            (1, 4),  # batch_size=1, features=4
            args.output_dir,
            f"{args.model_type}_asset_tracking"
        )
    
    logger.info(f"Models exported to: {exported_paths}")
    
    # Create deployment manager
    deployment_manager = create_deployment_manager(config)
    
    # Simulate deployments to different devices
    device_configs = {
        "raspberry_pi_4": {
            "device_type": "arm64",
            "target_runtime": "tflite",
            "memory_gb": 4,
            "power_consumption_w": 3.4
        },
        "jetson_nano": {
            "device_type": "arm64", 
            "target_runtime": "onnx",
            "memory_gb": 4,
            "power_consumption_w": 5.0
        },
        "android_mobile": {
            "device_type": "arm64",
            "target_runtime": "tflite", 
            "memory_gb": 6,
            "power_consumption_w": 2.0
        }
    }
    
    logger.info("Simulating deployments to edge devices...")
    
    for device_name, device_config in device_configs.items():
        deployment_id = f"{args.model_type}_{device_name}"
        
        # Choose appropriate model format
        if device_config["target_runtime"] == "tflite":
            model_path = exported_paths["tflite"]
        elif device_config["target_runtime"] == "onnx":
            model_path = exported_paths["onnx"]
        else:
            model_path = exported_paths["onnx"]  # Default
        
        # Deploy model
        deployment_info = deployment_manager.deploy_to_device(
            model_path, device_config, deployment_id
        )
        
        logger.info(f"Deployed to {device_name}: {deployment_info['status']}")
    
    # List all deployments
    deployments = deployment_manager.list_deployments()
    logger.info(f"Total deployments: {len(deployments)}")
    
    logger.info("Deployment completed successfully!")


if __name__ == "__main__":
    main()
