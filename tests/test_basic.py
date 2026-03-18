"""Asset Tracking System - Basic Tests."""

import pytest
import torch
import numpy as np

from src.models import create_model, count_parameters
from src.pipelines.data_pipeline import AssetDataGenerator, AssetDataProcessor
from src.utils import set_random_seeds, get_device


class TestModels:
    """Test model functionality."""
    
    def test_baseline_model_creation(self):
        """Test baseline model creation."""
        model = create_model("baseline")
        assert isinstance(model, torch.nn.Module)
        assert count_parameters(model) > 0
    
    def test_edge_model_creation(self):
        """Test edge model creation."""
        model = create_model("edge")
        assert isinstance(model, torch.nn.Module)
        assert count_parameters(model) > 0
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = create_model("baseline")
        x = torch.randn(1, 4)
        output = model(x)
        assert output.shape == (1, 2)
    
    def test_model_prediction(self):
        """Test model prediction methods."""
        model = create_model("edge")
        x = torch.randn(5, 4)
        
        predictions = model.predict(x)
        probabilities = model.predict_proba(x)
        
        assert predictions.shape == (5,)
        assert probabilities.shape == (5, 2)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6)


class TestDataPipeline:
    """Test data pipeline functionality."""
    
    def test_data_generator(self):
        """Test data generation."""
        generator = AssetDataGenerator(random_seed=42)
        features, labels = generator.generate_dataset(100)
        
        assert features.shape == (100, 4)
        assert labels.shape == (100,)
        assert np.all(np.isin(labels, [0, 1]))
    
    def test_data_processor(self):
        """Test data processing."""
        processor = AssetDataProcessor(random_seed=42)
        generator = AssetDataGenerator(random_seed=42)
        
        features, labels = generator.generate_dataset(100)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(features, labels)
        
        assert len(X_train) + len(X_val) + len(X_test) == 100
        assert len(y_train) + len(y_val) + len(y_test) == 100
    
    def test_data_scaling(self):
        """Test data scaling."""
        processor = AssetDataProcessor(random_seed=42)
        generator = AssetDataGenerator(random_seed=42)
        
        features, labels = generator.generate_dataset(100)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(features, labels)
        
        X_train_scaled = processor.fit_transform_features(X_train)
        X_val_scaled = processor.transform_features(X_val)
        
        assert X_train_scaled.shape == X_train.shape
        assert X_val_scaled.shape == X_val.shape


class TestUtils:
    """Test utility functions."""
    
    def test_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        
        # Test numpy
        np_val1 = np.random.random()
        set_random_seeds(42)
        np_val2 = np.random.random()
        assert np_val1 == np_val2
        
        # Test torch
        torch_val1 = torch.randn(1)
        set_random_seeds(42)
        torch_val2 = torch.randn(1)
        assert torch.allclose(torch_val1, torch_val2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__])
