import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, List

import fiftyone as fo
from fiftyone import Model

import depth_pro

def get_device() -> str:
    """Helper function to determine the best available device.
    
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
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
    """FiftyOne Model wrapper for Apple's DepthPro model."""

    def __init__(self, depth_type: str) -> None:
        """Initialize the model.
        
        Args:
            depth_type (str): Type of depth visualization ('inverse' or 'regular')
        """
        valid_types = {"inverse", "regular"}
        if depth_type not in valid_types:
            raise ValueError(f"depth_type must be one of {valid_types}")
        self.depth_type = depth_type
        
        self.device = get_device()
        self.model = None
        self.transform = None
        
        # Ensure model file exists
        self._download_model_if_needed()

    def _download_model_if_needed(self) -> None:
        """Downloads the DepthPro model if not already present."""
        try:
            os.makedirs("checkpoints", exist_ok=True)
            model_path = os.path.join("checkpoints", "depth_pro.pt")
            if not os.path.exists(model_path):
                print("Downloading DepthPro model...")
                result = os.system(f"huggingface-cli download --local-dir checkpoints apple/DepthPro")
                if result != 0:
                    raise RuntimeError("Model download failed")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")

    def _setup_model(self) -> None:
        """Initializes the DepthPro model and transform."""
        if self.model is None:
            self.model, self.transform = depth_pro.create_model_and_transforms(
                device=self.device,
                precision=torch.float16
            )
            self.model.eval()

    @property
    def media_type(self):
        return "image"

    def _process_depth_map(self, depth_np: np.ndarray) -> np.ndarray:
        """Process depth map based on visualization preference.
        
        Args:
            depth_np (np.ndarray): Raw depth map in meters
            
        Returns:
            np.ndarray: Normalized depth map as uint8 array (0-255)
            
        Note:
            For inverse depth (1/depth):
            - Better visualization of nearby objects
            - More detail in close range
            - Useful for visual SLAM or Structure from Motion
            - Better for indoor environments

            For regular depth:
            - Linear depth representation
            - Better for absolute distance measurements
            - Ideal for 3D reconstructions
            - Common in autonomous driving
        """
        if self.depth_type =="inverse":
            # Convert to inverse depth
            inverse_depth = 1.0 / depth_np
            
            # Normalize inverse depth to 0-255
            normalized_map = ((inverse_depth - np.min(inverse_depth)) / 
                            (np.max(inverse_depth) - np.min(inverse_depth)) * 255).astype("uint8")
            
        if self.depth_type =="regular":
            # Normalize regular depth to 0-255
            normalized_map = ((depth_np - np.min(depth_np)) / 
                            (np.max(depth_np) - np.min(depth_np)) * 255).astype("uint8")
            
        return normalized_map

    def _predict(self, image: Image.Image) -> Dict[str, fo.Heatmap]:
        """Perform depth prediction on a single image.

        Args:
            image (Image.Image): The input image

        Returns:
            Dict[str, fo.Heatmap]: Dictionary containing depth heatmap
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

    def predict(self, image: np.ndarray) -> Dict[str, fo.Heatmap]:
        """Predict depth for the given image.

        Args:
            image (np.ndarray): The input image as a numpy array

        Returns:
            Dict[str, fo.Heatmap]: Dictionary containing depth heatmap
        """
        image_pil = Image.fromarray(image)
        return self._predict(image_pil)

    def predict_all(self, images: List[np.ndarray]) -> List[Dict[str, fo.Heatmap]]:
        """Perform prediction on a list of images.

        Args:
            images (List[np.ndarray]): List of input images as numpy arrays

        Returns:
            List[Dict[str, fo.Heatmap]]: List of prediction dictionaries for each image
        """
        return [self.predict(image) for image in images]
    
    def __del__(self):
        """Cleanup when the model is deleted."""
        if hasattr(self, 'model') and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            torch.cuda.empty_cache()

def run_depth_prediction(
    dataset: fo.Dataset,
    depth_field: str,
    depth_type: str
) -> None:
    """Run depth prediction on a FiftyOne dataset.
    
    Args:
        dataset (fo.Dataset): FiftyOne dataset to process
        depth_field (str): Field name to store depth predictions
        depth_type (str): Type of depth visualization ('inverse' or 'regular')
    """
    model = DepthProModel(
        depth_type=depth_type
    )
    dataset.apply_model(model, label_field=depth_field)