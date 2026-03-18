"""Asset Tracking System - Evaluation Module.

This module provides comprehensive evaluation metrics and benchmarking
for asset tracking models, including accuracy and edge performance metrics.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


class AssetTrackingEvaluator:
    """Comprehensive evaluator for asset tracking models.
    
    Provides both accuracy metrics and edge performance metrics
    for model evaluation and comparison.
    """
    
    def __init__(self, device: torch.device) -> None:
        """Initialize the evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device
    
    def evaluate_accuracy(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate model accuracy metrics.
        
        Args:
            model: PyTorch model to evaluate
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing accuracy metrics
        """
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Get predictions
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                outputs = model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, all_predictions)
        precision = precision_score(y_test, all_predictions, average='weighted')
        recall = recall_score(y_test, all_predictions, average='weighted')
        f1 = f1_score(y_test, all_predictions, average='weighted')
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, all_probabilities[:, 1])
        else:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, all_predictions)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
            "confusion_matrix": cm.tolist(),
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC: {auc:.4f}")
        
        return metrics
    
    def evaluate_edge_performance(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        n_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """Evaluate edge performance metrics.
        
        Args:
            model: PyTorch model to evaluate
            X_test: Test features
            n_runs: Number of runs for timing
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary containing edge performance metrics
        """
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(X_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Timing runs
        times = []
        memory_usage = []
        
        with torch.no_grad():
            for _ in range(n_runs):
                # Memory before
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated(self.device)
                
                # Inference
                start_time = time.time()
                _ = model(X_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                
                # Memory after
                if self.device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated(self.device)
                    memory_usage.append(memory_after - memory_before)
                
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        latency_p50 = np.percentile(times, 50)
        latency_p95 = np.percentile(times, 95)
        latency_p99 = np.percentile(times, 99)
        
        throughput = len(X_test) / np.mean(times)
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        metrics = {
            "latency_p50": latency_p50,
            "latency_p95": latency_p95,
            "latency_p99": latency_p99,
            "throughput": throughput,
            "model_size_mb": model_size_mb,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
        
        if memory_usage:
            metrics["peak_memory_mb"] = max(memory_usage) / (1024 * 1024)
            metrics["avg_memory_mb"] = np.mean(memory_usage) / (1024 * 1024)
        
        logger.info(f"Latency P50: {latency_p50*1000:.2f}ms")
        logger.info(f"Latency P95: {latency_p95*1000:.2f}ms")
        logger.info(f"Throughput: {throughput:.2f} samples/sec")
        logger.info(f"Model Size: {model_size_mb:.2f} MB")
        
        return metrics
    
    def stress_test(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
    ) -> Dict[str, Dict[str, float]]:
        """Perform stress testing with different noise levels.
        
        Args:
            model: PyTorch model to test
            X_test: Test features
            y_test: Test labels
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary containing stress test results
        """
        results = {}
        
        for noise_level in noise_levels:
            logger.info(f"Stress testing with noise level: {noise_level}")
            
            # Add noise to test data
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise
            
            # Evaluate accuracy
            accuracy_metrics = self.evaluate_accuracy(model, X_noisy, y_test)
            
            results[f"noise_{noise_level}"] = accuracy_metrics
        
        return results
    
    def comprehensive_evaluation(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        stress_test_config: Optional[Dict] = None,
    ) -> Dict[str, Union[Dict, float]]:
        """Perform comprehensive evaluation including all metrics.
        
        Args:
            model: PyTorch model to evaluate
            X_test: Test features
            y_test: Test labels
            stress_test_config: Configuration for stress testing
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        
        results = {}
        
        # Accuracy evaluation
        logger.info("Evaluating accuracy metrics...")
        results["accuracy"] = self.evaluate_accuracy(model, X_test, y_test)
        
        # Edge performance evaluation
        logger.info("Evaluating edge performance metrics...")
        results["edge_performance"] = self.evaluate_edge_performance(model, X_test)
        
        # Stress testing
        if stress_test_config:
            logger.info("Performing stress testing...")
            stress_config = stress_test_config.get("stress_test", {})
            noise_levels = stress_config.get("noise_levels", [0.0, 0.1, 0.2, 0.3])
            results["stress_test"] = self.stress_test(model, X_test, y_test, noise_levels)
        
        logger.info("Comprehensive evaluation completed")
        
        return results


class ModelComparison:
    """Utility class for comparing multiple models.
    
    Provides methods to compare different models and create
    performance leaderboards.
    """
    
    def __init__(self, evaluator: AssetTrackingEvaluator) -> None:
        """Initialize the model comparison.
        
        Args:
            evaluator: Evaluator instance to use
        """
        self.evaluator = evaluator
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        X_test: np.ndarray,
        y_test: np.ndarray,
        stress_test_config: Optional[Dict] = None,
    ) -> Dict[str, Dict[str, Union[Dict, float]]]:
        """Compare multiple models comprehensively.
        
        Args:
            models: Dictionary of model names to models
            X_test: Test features
            y_test: Test labels
            stress_test_config: Configuration for stress testing
            
        Returns:
            Dictionary containing comparison results
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            results[name] = self.evaluator.comprehensive_evaluation(
                model, X_test, y_test, stress_test_config
            )
        
        return results
    
    def create_leaderboard(
        self,
        comparison_results: Dict[str, Dict[str, Union[Dict, float]]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Create performance leaderboard from comparison results.
        
        Args:
            comparison_results: Results from compare_models
            
        Returns:
            Dictionary containing leaderboards for different metrics
        """
        leaderboards = {
            "accuracy": [],
            "f1_score": [],
            "latency_p50": [],
            "throughput": [],
            "model_size_mb": [],
        }
        
        for model_name, results in comparison_results.items():
            # Accuracy metrics
            accuracy = results["accuracy"]["accuracy"]
            f1 = results["accuracy"]["f1_score"]
            
            # Edge performance metrics
            latency_p50 = results["edge_performance"]["latency_p50"]
            throughput = results["edge_performance"]["throughput"]
            model_size = results["edge_performance"]["model_size_mb"]
            
            leaderboards["accuracy"].append((model_name, accuracy))
            leaderboards["f1_score"].append((model_name, f1))
            leaderboards["latency_p50"].append((model_name, latency_p50))
            leaderboards["throughput"].append((model_name, throughput))
            leaderboards["model_size_mb"].append((model_name, model_size))
        
        # Sort leaderboards
        for metric in leaderboards:
            if metric in ["latency_p50", "model_size_mb"]:
                # Lower is better
                leaderboards[metric].sort(key=lambda x: x[1])
            else:
                # Higher is better
                leaderboards[metric].sort(key=lambda x: x[1], reverse=True)
        
        return leaderboards
    
    def print_leaderboard(
        self,
        leaderboards: Dict[str, List[Tuple[str, float]]],
    ) -> None:
        """Print formatted leaderboard.
        
        Args:
            leaderboards: Leaderboards from create_leaderboard
        """
        print("\n" + "="*80)
        print("ASSET TRACKING MODEL LEADERBOARD")
        print("="*80)
        
        for metric, rankings in leaderboards.items():
            print(f"\n{metric.upper().replace('_', ' ')}:")
            print("-" * 40)
            for i, (model_name, score) in enumerate(rankings, 1):
                if metric in ["latency_p50"]:
                    print(f"{i:2d}. {model_name:20s}: {score*1000:8.2f} ms")
                elif metric in ["model_size_mb"]:
                    print(f"{i:2d}. {model_name:20s}: {score:8.2f} MB")
                elif metric in ["throughput"]:
                    print(f"{i:2d}. {model_name:20s}: {score:8.2f} samples/sec")
                else:
                    print(f"{i:2d}. {model_name:20s}: {score:8.4f}")
        
        print("\n" + "="*80)


def create_evaluator(device: torch.device) -> AssetTrackingEvaluator:
    """Factory function to create an evaluator.
    
    Args:
        device: Device to run evaluation on
        
    Returns:
        Configured evaluator instance
    """
    return AssetTrackingEvaluator(device)
