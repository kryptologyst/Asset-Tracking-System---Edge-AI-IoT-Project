"""Asset Tracking System - Core Models Module.

This module contains the neural network models for asset tracking classification,
including baseline and edge-optimized variants with compression techniques.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import quantize_dynamic
from torch_pruning import magnitude_pruning


class AssetTrackingBaseline(nn.Module):
    """Baseline neural network for asset tracking classification.
    
    A simple feedforward network that classifies assets as "Normal" or "Lost"
    based on sensor features: location variance, signal strength, days inactive,
    and motion count.
    
    Args:
        input_features: Number of input features (default: 4)
        hidden_sizes: List of hidden layer sizes (default: [32, 16])
        dropout_rate: Dropout rate for regularization (default: 0.2)
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        input_features: int = 4,
        hidden_sizes: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        
        self.input_features = input_features
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_size = input_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Predicted class labels of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class AssetTrackingEdgeOptimized(nn.Module):
    """Edge-optimized neural network for asset tracking classification.
    
    A lightweight network designed for deployment on edge devices with
    reduced computational requirements while maintaining accuracy.
    
    Args:
        input_features: Number of input features (default: 4)
        hidden_size: Single hidden layer size (default: 16)
        dropout_rate: Dropout rate for regularization (default: 0.1)
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        input_features: int = 4,
        hidden_size: int = 16,
        dropout_rate: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # Simplified architecture for edge deployment
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.
        
        Args:
            x: Input tensor of shape (batch_size, input_features)
            
        Returns:
            Predicted class labels of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class ModelCompressor:
    """Model compression utilities for edge deployment.
    
    Provides methods for quantization, pruning, and other compression techniques
    to optimize models for edge deployment.
    """
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        method: str = "dynamic",
        bits: int = 8,
    ) -> nn.Module:
        """Quantize model for reduced precision inference.
        
        Args:
            model: PyTorch model to quantize
            method: Quantization method ("dynamic", "static", "qat")
            bits: Number of bits for quantization (8 or 4)
            
        Returns:
            Quantized model
        """
        if method == "dynamic":
            if bits == 8:
                return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            elif bits == 4:
                return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint4)
            else:
                raise ValueError(f"Unsupported bit width: {bits}")
        else:
            raise NotImplementedError(f"Quantization method {method} not implemented")
    
    @staticmethod
    def prune_model(
        model: nn.Module,
        sparsity: float = 0.3,
        method: str = "magnitude",
    ) -> nn.Module:
        """Prune model to reduce parameters.
        
        Args:
            model: PyTorch model to prune
            sparsity: Target sparsity ratio (0.0 to 1.0)
            method: Pruning method ("magnitude", "structured", "channel")
            
        Returns:
            Pruned model
        """
        if method == "magnitude":
            # Create a copy to avoid modifying the original
            pruned_model = type(model)(**model.__dict__)
            pruned_model.load_state_dict(model.state_dict())
            
            # Apply magnitude pruning
            magnitude_pruning(pruned_model, sparsity)
            return pruned_model
        else:
            raise NotImplementedError(f"Pruning method {method} not implemented")
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Calculate model size metrics.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing size metrics in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            "parameters_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_mb": total_size / (1024 * 1024),
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }


def create_model(
    model_type: str = "baseline",
    **kwargs,
) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ("baseline" or "edge")
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Instantiated model
    """
    if model_type == "baseline":
        return AssetTrackingBaseline(**kwargs)
    elif model_type == "edge":
        return AssetTrackingEdgeOptimized(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
