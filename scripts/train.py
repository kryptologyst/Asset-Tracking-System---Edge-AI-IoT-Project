#!/usr/bin/env python3
"""Asset Tracking System - Main Training Script.

This script trains baseline and edge-optimized models for asset tracking,
performs comprehensive evaluation, and exports models for edge deployment.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.asset_tracking_models import create_model, ModelCompressor
from pipelines.data_pipeline import create_data_pipeline
from pipelines.training_pipeline import create_trainer, ModelOptimizer
from pipelines.evaluation_pipeline import create_evaluator, ModelComparison
from export.model_export import create_exporter, create_deployment_manager


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/training.log"),
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Asset Tracking Models")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--device", default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config["training"]["random_seed"])
    
    # Create data pipeline
    logger.info("Creating data pipeline...")
    data_generator, data_processor, data_streamer = create_data_pipeline(config)
    
    # Generate dataset
    logger.info("Generating dataset...")
    X, y = data_generator.generate_dataset(config["data"]["n_samples"])
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X, y,
        test_size=config["data"]["test_split"],
        val_size=config["data"]["val_split"]
    )
    
    # Scale features
    X_train_scaled = data_processor.fit_transform_features(X_train)
    X_val_scaled = data_processor.transform_features(X_val)
    X_test_scaled = data_processor.transform_features(X_test)
    
    # Train baseline model
    logger.info("Training baseline model...")
    baseline_trainer = create_trainer("baseline", device, config["training"])
    baseline_history = baseline_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Train edge model
    logger.info("Training edge-optimized model...")
    edge_trainer = create_trainer("edge", device, config["training"])
    edge_history = edge_trainer.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = create_evaluator(device)
    
    baseline_metrics = evaluator.comprehensive_evaluation(
        baseline_trainer.model, X_test_scaled, y_test, config["evaluation"]
    )
    edge_metrics = evaluator.comprehensive_evaluation(
        edge_trainer.model, X_test_scaled, y_test, config["evaluation"]
    )
    
    # Compare models
    models = {
        "baseline": baseline_trainer.model,
        "edge": edge_trainer.model,
    }
    
    comparison = ModelComparison(evaluator)
    comparison_results = comparison.compare_models(models, X_test_scaled, y_test, config["evaluation"])
    leaderboards = comparison.create_leaderboard(comparison_results)
    comparison.print_leaderboard(leaderboards)
    
    # Export models
    logger.info("Exporting models...")
    exporter = create_exporter(device)
    
    os.makedirs("assets/models", exist_ok=True)
    
    # Export baseline model
    baseline_paths = exporter.export_all_formats(
        baseline_trainer.model,
        (1, 4),  # batch_size=1, features=4
        "assets/models",
        "baseline"
    )
    
    # Export edge model
    edge_paths = exporter.export_all_formats(
        edge_trainer.model,
        (1, 4),
        "assets/models",
        "edge"
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Baseline model exported to: {baseline_paths}")
    logger.info(f"Edge model exported to: {edge_paths}")


if __name__ == "__main__":
    main()
