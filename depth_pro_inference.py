import os
import torch
import numpy as np
from PIL import Image
from typing import Dict

import fiftyone as fo
from fiftyone import Model

import depth_pro

def get_device():
    """Helper function to determine the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) device")
    else:
        device = "cpu"
        print("Using CPU device")
    return device

class DepthProModel(Model):
    """
    FiftyOne Model wrapper for Apple's DepthPro model.
    """

    def __init__(self, inverse_depth):
        """
        Initialize the model.
        
        Args:
            checkpoint_dir (str): Directory where the model checkpoints are stored
            inverse_depth (bool): If True, use inverse depth for visualization
        """
        self.device = get_device()
        self.model = None
        self.transform = None
        self.inverse_depth = inverse_depth
        
        # Download and initialize the model
        self._setup_model()

    def _setup_model(self):
        """Downloads and initializes the DepthPro model."""
        os.makedirs("checkpoints", exist_ok=True)
        
        if not os.path.exists(os.path.join("checkpoints", "depth_pro.pt")):
            print("Downloading DepthPro model...")
            os.system(f"huggingface-cli download --local-dir checkpoints apple/DepthPro")
            
        if self.model is None:
            self.model, self.transform = depth_pro.create_model_and_transforms(
                device=self.device,
                precision=torch.float16
            )
            self.model.eval()

    @property
    def media_type(self):
        return "image"

    def _process_depth_map(self, depth_np: np.ndarray) -> tuple:
        """
        Process depth map based on visualization preference.
        
        For inverse depth (1/depth):
        - Better visualization of nearby objects
        - More detail in close range
        - Doing visual SLAM or Structure from Motion
        - Visualizing indoor environments

        For regular depth:
        - Linear depth representation
        - Better for absolute distance measurements
        - Creating 3D reconstructions
        - Common in autonomous driving
        
        Args:
            depth_np (np.ndarray): Raw depth map in meters
            
        Returns:
            tuple: (normalized_map, metadata_dict)
        """
        if self.inverse_depth:
            # Convert to inverse depth
            inverse_depth = 1.0 / depth_np
            
            # Normalize inverse depth to 0-255
            normalized_map = ((inverse_depth - np.min(inverse_depth)) / 
                            (np.max(inverse_depth) - np.min(inverse_depth)) * 255).astype("uint8")
            
        else:
            # Normalize regular depth to 0-255
            normalized_map = ((depth_np - np.min(depth_np)) / 
                            (np.max(depth_np) - np.min(depth_np)) * 255).astype("uint8")
            
        return normalized_map

    def _predict(self, image: Image.Image) -> Dict:
        """
        Performs depth prediction on a single image.

        Args:
            image (PIL.Image.Image): The input image

        Returns:
            Dict: Contains depth heatmap and metadata
        """
        if self.model is None:
            self._setup_model()

        image_tensor = self.transform(image)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            prediction = self.model.infer(image_tensor, f_px=None)
            depth_meters = prediction["depth"]

            # Convert to numpy
            depth_np = depth_meters.cpu().numpy()

            # Process depth map based on user preference
            normalized_map = self._process_depth_map(depth_np)
            
            result = {
                "depth": fo.Heatmap(map=normalized_map),
            }

            del image_tensor
            torch.cuda.empty_cache()

            return result

    def predict(self, image: np.ndarray) -> Dict:
        """
        Predicts depth for the given image.

        Args:
            image (np.ndarray): The input image as a numpy array

        Returns:
            Dict: Contains depth heatmap and metadata
        """
        image_pil = Image.fromarray(image)
        return self._predict(image_pil)
    
    def __del__(self):
        """Cleanup when the model is deleted."""
        if hasattr(self, 'model') and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            torch.cuda.empty_cache()

def run_depth_prediction(
    dataset, 
    depth_field, 
    inverse_depth
):
    """
    Runs depth prediction on a FiftyOne dataset.
    
    Args:
        dataset: FiftyOne dataset
        depth_field (str): Field name to store depth predictions
        checkpoint_dir (str): Directory to store model checkpoints
        inverse_depth (bool): If True, use inverse depth for visualization
    """
    model = DepthProModel(
        inverse_depth=inverse_depth
    )
    dataset.apply_model(model, label_field=depth_field)