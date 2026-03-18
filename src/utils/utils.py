"""Asset Tracking System - Utility Functions.

Common utility functions for logging, device management, and data processing.
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Optional custom log format
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = "auto") -> torch.device:
    """Get PyTorch device based on availability and preference.
    
    Args:
        device_name: Device name ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    
    return device


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_size(bytes_size: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def save_dict_to_yaml(data: Dict[str, Any], file_path: str) -> None:
    """Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save YAML file
    """
    import yaml
    
    create_directory(os.path.dirname(file_path))
    
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def load_yaml_to_dict(file_path: str) -> Dict[str, Any]:
    """Load YAML file to dictionary.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary loaded from YAML
    """
    import yaml
    
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and print result."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} completed in {format_time(duration)}")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    
    return True


def print_model_summary(model: torch.nn.Module) -> None:
    """Print model summary with parameter count and size.
    
    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = param_size + buffer_size
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {format_size(model_size)}")
    print(f"  Model architecture: {model.__class__.__name__}")


def calculate_model_flops(model: torch.nn.Module, input_shape: tuple) -> int:
    """Calculate approximate FLOPs for a model.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        
    Returns:
        Approximate FLOPs count
    """
    flops = 0
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # FLOPs = 2 * input_features * output_features (multiply-add operations)
            flops += 2 * module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv2d):
            # Approximate FLOPs for Conv2d
            kernel_flops = module.kernel_size[0] * module.kernel_size[1]
            flops += 2 * kernel_flops * module.in_channels * module.out_channels
    
    return flops


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple,
    device: torch.device,
    n_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark model inference performance.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark results
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        "mean_latency": np.mean(times),
        "std_latency": np.std(times),
        "min_latency": np.min(times),
        "max_latency": np.max(times),
        "p50_latency": np.percentile(times, 50),
        "p95_latency": np.percentile(times, 95),
        "p99_latency": np.percentile(times, 99),
        "throughput": input_shape[0] / np.mean(times),
    }
