"""Asset Tracking System - Data Pipeline Module.

This module handles data generation, preprocessing, and streaming simulation
for the asset tracking system.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AssetDataGenerator:
    """Generates synthetic asset tracking data for training and testing.
    
    Creates realistic sensor data including location variance, signal strength,
    days inactive, and motion count based on configurable parameters.
    """
    
    def __init__(
        self,
        random_seed: int = 42,
        config: Optional[Dict] = None,
    ) -> None:
        """Initialize the data generator.
        
        Args:
            random_seed: Random seed for reproducibility
            config: Configuration dictionary for data generation parameters
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Default configuration
        self.config = config or {
            "location_variance": {"mean": 5.0, "std": 2.0},
            "signal_strength": {"mean": 70.0, "std": 10.0},
            "days_inactive": {"min": 0, "max": 30},
            "motion_count": {"lambda": 3.0},
        }
        
    def generate_features(
        self,
        n_samples: int,
        noise_level: float = 0.0,
    ) -> np.ndarray:
        """Generate synthetic sensor features.
        
        Args:
            n_samples: Number of samples to generate
            noise_level: Amount of noise to add (0.0 to 1.0)
            
        Returns:
            Feature matrix of shape (n_samples, 4)
        """
        # Location variance (low variance indicates stuck/lost assets)
        location_variance = np.random.normal(
            self.config["location_variance"]["mean"],
            self.config["location_variance"]["std"],
            n_samples
        )
        
        # Signal strength (RSSI in dBm)
        signal_strength = np.random.normal(
            self.config["signal_strength"]["mean"],
            self.config["signal_strength"]["std"],
            n_samples
        )
        
        # Days inactive (higher values indicate suspicious behavior)
        days_inactive = np.random.randint(
            self.config["days_inactive"]["min"],
            self.config["days_inactive"]["max"] + 1,
            n_samples
        )
        
        # Motion count (Poisson distribution for discrete events)
        motion_count = np.random.poisson(
            self.config["motion_count"]["lambda"],
            n_samples
        )
        
        # Stack features
        features = np.stack([
            location_variance,
            signal_strength,
            days_inactive,
            motion_count
        ], axis=1)
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, features.shape)
            features = features + noise
            
        return features
    
    def generate_labels(
        self,
        features: np.ndarray,
        threshold_config: Optional[Dict] = None,
    ) -> np.ndarray:
        """Generate binary labels based on feature patterns.
        
        Args:
            features: Feature matrix of shape (n_samples, 4)
            threshold_config: Custom thresholds for label generation
            
        Returns:
            Binary labels (0 = normal, 1 = lost/suspicious)
        """
        # Default thresholds
        thresholds = threshold_config or {
            "location_variance_max": 3.0,
            "signal_strength_min": 60.0,
            "days_inactive_min": 10,
        }
        
        location_variance, signal_strength, days_inactive, motion_count = features.T
        
        # Create labels based on suspicious patterns
        lost_conditions = (
            (location_variance < thresholds["location_variance_max"]) &
            (signal_strength < thresholds["signal_strength_min"]) &
            (days_inactive > thresholds["days_inactive_min"])
        )
        
        return lost_conditions.astype(int)
    
    def generate_dataset(
        self,
        n_samples: int,
        noise_level: float = 0.0,
        threshold_config: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete dataset with features and labels.
        
        Args:
            n_samples: Number of samples to generate
            noise_level: Amount of noise to add
            threshold_config: Custom thresholds for label generation
            
        Returns:
            Tuple of (features, labels)
        """
        features = self.generate_features(n_samples, noise_level)
        labels = self.generate_labels(features, threshold_config)
        
        logger.info(f"Generated dataset with {n_samples} samples")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return features, labels


class AssetDataProcessor:
    """Processes and preprocesses asset tracking data.
    
    Handles data splitting, scaling, and preparation for model training.
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize the data processor.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.
        
        Args:
            features: Feature matrix
            labels: Label vector
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            stratify: Whether to stratify splits by class
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        stratify_labels = labels if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=stratify_temp
        )
        
        logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_scaler(self, X_train: np.ndarray) -> None:
        """Fit the scaler on training data.
        
        Args:
            X_train: Training features
        """
        self.scaler.fit(X_train)
        self.is_fitted = True
        logger.info("Scaler fitted on training data")
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")
        
        return self.scaler.transform(X)
    
    def fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform features in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        return self.scaler.fit_transform(X)
    
    def inverse_transform_features(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform features to original scale.
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Original scale feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        return self.scaler.inverse_transform(X)


class AssetDataStreamer:
    """Simulates real-time streaming of asset tracking data.
    
    Provides a streaming interface for continuous data generation
    and processing in edge deployment scenarios.
    """
    
    def __init__(
        self,
        data_generator: AssetDataGenerator,
        data_processor: AssetDataProcessor,
        update_interval: float = 1.0,
    ) -> None:
        """Initialize the data streamer.
        
        Args:
            data_generator: Data generator instance
            data_processor: Data processor instance
            update_interval: Time interval between updates in seconds
        """
        self.data_generator = data_generator
        self.data_processor = data_processor
        self.update_interval = update_interval
        self.current_features = None
        self.current_labels = None
        
    def generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single sample of asset data.
        
        Returns:
            Tuple of (features, labels) for one asset
        """
        features, labels = self.data_generator.generate_dataset(1)
        return features[0], labels[0]
    
    def stream_data(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Stream data samples one by one.
        
        Args:
            n_samples: Number of samples to stream
            
        Yields:
            Tuple of (features, labels) for each sample
        """
        for _ in range(n_samples):
            features, labels = self.generate_sample()
            yield features, labels
    
    def get_current_state(self) -> Dict[str, Union[np.ndarray, None]]:
        """Get current asset state.
        
        Returns:
            Dictionary containing current features and labels
        """
        return {
            "features": self.current_features,
            "labels": self.current_labels,
        }
    
    def update_state(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Update current asset state.
        
        Args:
            features: New feature vector
            labels: New label
        """
        self.current_features = features
        self.current_labels = labels


def create_data_pipeline(
    config: Dict,
    random_seed: int = 42,
) -> Tuple[AssetDataGenerator, AssetDataProcessor, AssetDataStreamer]:
    """Create a complete data pipeline.
    
    Args:
        config: Configuration dictionary
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data_generator, data_processor, data_streamer)
    """
    # Create components
    data_generator = AssetDataGenerator(random_seed, config.get("synthetic_data"))
    data_processor = AssetDataProcessor(random_seed)
    
    # Create streamer with update interval from config
    update_interval = config.get("demo", {}).get("simulation", {}).get("update_interval", 1.0)
    data_streamer = AssetDataStreamer(data_generator, data_processor, update_interval)
    
    return data_generator, data_processor, data_streamer
