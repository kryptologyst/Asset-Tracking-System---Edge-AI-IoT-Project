"""Asset Tracking System - Export and Deployment Module.

This module handles model export to various edge runtime formats
and deployment utilities for edge devices.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import onnx
import tensorflow as tf
from coremltools import convert

logger = logging.getLogger(__name__)


class ModelExporter:
    """Exports PyTorch models to various edge runtime formats.
    
    Supports export to ONNX, TensorFlow Lite, CoreML, and OpenVINO formats
    for deployment on different edge devices.
    """
    
    def __init__(self, device: torch.device) -> None:
        """Initialize the model exporter.
        
        Args:
            device: Device to run export on
        """
        self.device = device
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 11,
    ) -> str:
        """Export model to ONNX format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape (batch_size, features)
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting model to ONNX format: {output_path}")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape[1:]).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info("ONNX export completed successfully")
        return output_path
    
    def export_to_tflite(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        quantize: bool = True,
    ) -> str:
        """Export model to TensorFlow Lite format.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save TFLite model
            quantize: Whether to apply quantization
            
        Returns:
            Path to exported TFLite model
        """
        logger.info(f"Exporting model to TFLite format: {output_path}")
        
        # First export to ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        self.export_to_onnx(model, input_shape, onnx_path)
        
        # Convert ONNX to TensorFlow
        tf_model = tf.keras.models.load_model(onnx_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info("TFLite export completed successfully")
        return output_path
    
    def export_to_coreml(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        quantize: bool = True,
    ) -> str:
        """Export model to CoreML format for iOS deployment.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_path: Path to save CoreML model
            quantize: Whether to apply quantization
            
        Returns:
            Path to exported CoreML model
        """
        logger.info(f"Exporting model to CoreML format: {output_path}")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape[1:]).to(self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_model = convert(
            traced_model,
            inputs=[('input', dummy_input)],
            outputs=['output'],
            minimum_deployment_target='13.0'
        )
        
        # Add metadata
        coreml_model.short_description = "Asset Tracking Classification Model"
        coreml_model.author = "Edge AI Research Team"
        coreml_model.license = "MIT"
        
        # Save CoreML model
        coreml_model.save(output_path)
        
        logger.info("CoreML export completed successfully")
        return output_path
    
    def export_all_formats(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_dir: str,
        model_name: str = "asset_tracking_model",
    ) -> Dict[str, str]:
        """Export model to all supported formats.
        
        Args:
            model: PyTorch model to export
            input_shape: Input tensor shape
            output_dir: Directory to save exported models
            model_name: Base name for exported models
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        exported_paths = {}
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        exported_paths["onnx"] = self.export_to_onnx(model, input_shape, onnx_path)
        
        # Export to TFLite
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        exported_paths["tflite"] = self.export_to_tflite(model, input_shape, tflite_path)
        
        # Export to CoreML
        coreml_path = os.path.join(output_dir, f"{model_name}.mlmodel")
        exported_paths["coreml"] = self.export_to_coreml(model, input_shape, coreml_path)
        
        logger.info(f"All formats exported to: {output_dir}")
        return exported_paths


class EdgeDeploymentManager:
    """Manages deployment of models to edge devices.
    
    Provides utilities for device-specific deployment,
    performance monitoring, and OTA updates.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize the deployment manager.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.deployed_models = {}
    
    def deploy_to_device(
        self,
        model_path: str,
        device_config: Dict,
        deployment_id: str,
    ) -> Dict[str, Union[str, bool]]:
        """Deploy model to a specific edge device.
        
        Args:
            model_path: Path to the model file
            device_config: Device-specific configuration
            deployment_id: Unique deployment identifier
            
        Returns:
            Dictionary containing deployment status
        """
        logger.info(f"Deploying model to device: {device_config.get('device_type', 'unknown')}")
        
        device_type = device_config.get("device_type", "unknown")
        target_runtime = device_config.get("target_runtime", "onnx")
        
        # Simulate deployment process
        deployment_info = {
            "deployment_id": deployment_id,
            "device_type": device_type,
            "target_runtime": target_runtime,
            "model_path": model_path,
            "deployment_time": "2024-01-01T00:00:00Z",  # Simulated
            "status": "deployed",
            "version": "1.0.0",
        }
        
        self.deployed_models[deployment_id] = deployment_info
        
        logger.info(f"Model deployed successfully: {deployment_id}")
        return deployment_info
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict]:
        """Get status of a deployment.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Deployment status information
        """
        return self.deployed_models.get(deployment_id)
    
    def list_deployments(self) -> List[Dict]:
        """List all deployments.
        
        Returns:
            List of deployment information
        """
        return list(self.deployed_models.values())
    
    def update_deployment(
        self,
        deployment_id: str,
        new_model_path: str,
        canary_percentage: float = 0.1,
    ) -> Dict[str, Union[str, bool]]:
        """Update a deployed model with OTA update.
        
        Args:
            deployment_id: Deployment identifier
            new_model_path: Path to new model
            canary_percentage: Percentage of devices for canary rollout
            
        Returns:
            Update status information
        """
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        logger.info(f"Updating deployment {deployment_id} with canary percentage {canary_percentage}")
        
        # Simulate OTA update
        update_info = {
            "deployment_id": deployment_id,
            "new_model_path": new_model_path,
            "canary_percentage": canary_percentage,
            "update_time": "2024-01-01T00:00:00Z",  # Simulated
            "status": "updating",
            "rollback_available": True,
        }
        
        # Update deployment info
        self.deployed_models[deployment_id]["status"] = "updating"
        self.deployed_models[deployment_id]["version"] = "1.1.0"
        
        logger.info(f"Deployment update initiated: {deployment_id}")
        return update_info
    
    def rollback_deployment(self, deployment_id: str) -> Dict[str, Union[str, bool]]:
        """Rollback a deployment to previous version.
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Rollback status information
        """
        if deployment_id not in self.deployed_models:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        logger.info(f"Rolling back deployment {deployment_id}")
        
        # Simulate rollback
        rollback_info = {
            "deployment_id": deployment_id,
            "rollback_time": "2024-01-01T00:00:00Z",  # Simulated
            "status": "rolled_back",
            "version": "1.0.0",  # Previous version
        }
        
        # Update deployment info
        self.deployed_models[deployment_id]["status"] = "deployed"
        self.deployed_models[deployment_id]["version"] = "1.0.0"
        
        logger.info(f"Deployment rolled back successfully: {deployment_id}")
        return rollback_info


class EdgeRuntimeManager:
    """Manages different edge runtime environments.
    
    Provides utilities for loading and running models
    on different edge runtime platforms.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize the runtime manager.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self.runtimes = {}
    
    def load_onnx_runtime(self, model_path: str) -> Dict[str, Union[str, bool]]:
        """Load model in ONNX Runtime.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Runtime information
        """
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            
            runtime_info = {
                "runtime_type": "onnx",
                "model_path": model_path,
                "providers": session.get_providers(),
                "input_names": [input.name for input in session.get_inputs()],
                "output_names": [output.name for output in session.get_outputs()],
                "status": "loaded",
            }
            
            self.runtimes["onnx"] = runtime_info
            logger.info("ONNX Runtime loaded successfully")
            
        except ImportError:
            logger.warning("ONNX Runtime not available")
            runtime_info = {"status": "not_available"}
        
        return runtime_info
    
    def load_tflite_runtime(self, model_path: str) -> Dict[str, Union[str, bool]]:
        """Load model in TensorFlow Lite Runtime.
        
        Args:
            model_path: Path to TFLite model
            
        Returns:
            Runtime information
        """
        try:
            import tflite_runtime.interpreter as tflite
            
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            runtime_info = {
                "runtime_type": "tflite",
                "model_path": model_path,
                "input_details": input_details,
                "output_details": output_details,
                "status": "loaded",
            }
            
            self.runtimes["tflite"] = runtime_info
            logger.info("TFLite Runtime loaded successfully")
            
        except ImportError:
            logger.warning("TFLite Runtime not available")
            runtime_info = {"status": "not_available"}
        
        return runtime_info
    
    def benchmark_runtime(
        self,
        runtime_type: str,
        input_data: np.ndarray,
        n_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark runtime performance.
        
        Args:
            runtime_type: Type of runtime ("onnx", "tflite")
            input_data: Input data for benchmarking
            n_runs: Number of runs for averaging
            
        Returns:
            Benchmark results
        """
        if runtime_type not in self.runtimes:
            raise ValueError(f"Runtime {runtime_type} not loaded")
        
        runtime_info = self.runtimes[runtime_type]
        
        if runtime_info["status"] != "loaded":
            raise ValueError(f"Runtime {runtime_type} not available")
        
        times = []
        
        if runtime_type == "onnx":
            import onnxruntime as ort
            session = ort.InferenceSession(runtime_info["model_path"])
            
            for _ in range(n_runs):
                start_time = time.time()
                session.run(None, {runtime_info["input_names"][0]: input_data})
                end_time = time.time()
                times.append(end_time - start_time)
        
        elif runtime_type == "tflite":
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=runtime_info["model_path"])
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            for _ in range(n_runs):
                start_time = time.time()
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                end_time = time.time()
                times.append(end_time - start_time)
        
        times = np.array(times)
        
        return {
            "avg_latency": np.mean(times),
            "std_latency": np.std(times),
            "p50_latency": np.percentile(times, 50),
            "p95_latency": np.percentile(times, 95),
            "throughput": len(input_data) / np.mean(times),
        }


def create_exporter(device: torch.device) -> ModelExporter:
    """Factory function to create a model exporter.
    
    Args:
        device: Device to run export on
        
    Returns:
        Configured exporter instance
    """
    return ModelExporter(device)


def create_deployment_manager(config: Dict) -> EdgeDeploymentManager:
    """Factory function to create a deployment manager.
    
    Args:
        config: Deployment configuration
        
    Returns:
        Configured deployment manager instance
    """
    return EdgeDeploymentManager(config)
