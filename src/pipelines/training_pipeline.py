"""Asset Tracking System - Training Pipeline Module.

This module handles model training, validation, and optimization for the
asset tracking system.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..models.asset_tracking_models import AssetTrackingBaseline, AssetTrackingEdgeOptimized, ModelCompressor

logger = logging.getLogger(__name__)


class AssetTrackingTrainer:
    """Trainer class for asset tracking models.
    
    Handles training, validation, and optimization of neural networks
    for asset tracking classification.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (cpu, cuda, etc.)
            config: Training configuration dictionary
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Training parameters
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.optimizer_name = config.get("optimizer", "adam")
        self.loss_function = config.get("loss_function", "binary_crossentropy")
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        
        # Initialize optimizer and loss function
        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            PyTorch optimizer
        """
        if self.optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration.
        
        Returns:
            PyTorch loss function
        """
        if self.loss_function == "binary_crossentropy":
            return nn.CrossEntropyLoss()
        elif self.loss_function == "mse":
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    def _create_data_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader from numpy arrays.
        
        Args:
            X: Feature matrix
            y: Label vector
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues
        )
        
        return dataloader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing training history
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create data loaders
        train_loader = self._create_data_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        
        # Training loop
        start_time = time.time()
        
        for epoch in tqdm(range(self.epochs), desc="Training"):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model state")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "training_time": training_time,
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        # Create test loader
        test_loader = self._create_data_loader(X_test, y_test, shuffle=False)
        
        # Evaluate
        test_loss, test_acc = self.validate_epoch(test_loader)
        
        # Additional metrics
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_test, all_predictions, average='weighted')
        recall = recall_score(y_test, all_predictions, average='weighted')
        f1 = f1_score(y_test, all_predictions, average='weighted')
        
        # ROC AUC (for binary classification)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, np.array(all_probabilities)[:, 1])
        else:
            auc = 0.0
        
        metrics = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc,
        }
        
        logger.info(f"Test Results: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        return metrics


class ModelOptimizer:
    """Model optimization utilities for edge deployment.
    
    Provides methods for model compression, quantization, and optimization
    to improve edge deployment performance.
    """
    
    def __init__(self, device: torch.device) -> None:
        """Initialize the model optimizer.
        
        Args:
            device: Device to perform optimization on
        """
        self.device = device
        self.compressor = ModelCompressor()
    
    def optimize_model(
        self,
        model: nn.Module,
        optimization_config: Dict,
    ) -> Dict[str, Union[nn.Module, Dict]]:
        """Apply various optimization techniques to a model.
        
        Args:
            model: PyTorch model to optimize
            optimization_config: Configuration for optimization techniques
            
        Returns:
            Dictionary containing optimized models and metrics
        """
        results = {"original_model": model}
        
        # Get original model size
        original_size = self.compressor.get_model_size(model)
        results["original_size"] = original_size
        
        # Quantization
        if optimization_config.get("quantization", {}).get("enabled", False):
            quant_config = optimization_config["quantization"]
            quantized_model = self.compressor.quantize_model(
                model,
                method=quant_config.get("method", "dynamic"),
                bits=quant_config.get("bits", 8)
            )
            results["quantized_model"] = quantized_model
            results["quantized_size"] = self.compressor.get_model_size(quantized_model)
        
        # Pruning
        if optimization_config.get("pruning", {}).get("enabled", False):
            prune_config = optimization_config["pruning"]
            pruned_model = self.compressor.prune_model(
                model,
                sparsity=prune_config.get("sparsity", 0.3),
                method=prune_config.get("method", "magnitude")
            )
            results["pruned_model"] = pruned_model
            results["pruned_size"] = self.compressor.get_model_size(pruned_model)
        
        return results
    
    def benchmark_models(
        self,
        models: Dict[str, nn.Module],
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_runs: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple models for performance comparison.
        
        Args:
            models: Dictionary of model names to models
            X_test: Test features
            y_test: Test labels
            n_runs: Number of runs for averaging
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {}
        
        for name, model in models.items():
            model = model.to(self.device)
            model.eval()
            
            # Convert test data to tensors
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_tensor = torch.LongTensor(y_test).to(self.device)
            
            # Benchmark inference time
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    start_time = time.time()
                    _ = model(X_tensor)
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            throughput = len(X_test) / avg_time
            
            # Model size
            model_size = self.compressor.get_model_size(model)
            
            # Accuracy
            with torch.no_grad():
                outputs = model(X_tensor)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y_tensor).float().mean().item()
            
            results[name] = {
                "avg_inference_time": avg_time,
                "std_inference_time": std_time,
                "throughput": throughput,
                "model_size_mb": model_size["total_mb"],
                "accuracy": accuracy,
            }
        
        return results


def create_trainer(
    model_type: str,
    device: torch.device,
    config: Dict,
) -> AssetTrackingTrainer:
    """Factory function to create a trainer.
    
    Args:
        model_type: Type of model ("baseline" or "edge")
        device: Device to train on
        config: Training configuration
        
    Returns:
        Configured trainer instance
    """
    from ..models.asset_tracking_models import create_model
    
    model = create_model(model_type)
    return AssetTrackingTrainer(model, device, config)
