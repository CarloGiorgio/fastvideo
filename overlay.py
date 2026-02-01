"""
Overlay Processors (Stub)
=========================

Classes for processing image overlays with velocity fields.
Full GPU-accelerated versions available in complete _video package.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class VelocityOverlayProcessor:
    """
    CPU-based velocity overlay processor.
    
    Stub implementation - provides basic functionality.
    """
    
    def __init__(self, 
                 base_preprocessor: Callable,
                 x: np.ndarray,
                 y: np.ndarray,
                 u: np.ndarray,
                 v: np.ndarray,
                 subsample: int = 2,
                 arrow_scale: float = 1.0,
                 arrow_color: Tuple[int, int, int] = (255, 255, 0),
                 arrow_width: int = 2):
        """
        Initialize velocity overlay processor.
        
        Parameters
        ----------
        base_preprocessor : callable
            Function to preprocess base image
        x, y : np.ndarray
            Coordinate meshgrids
        u, v : np.ndarray
            Velocity components
        subsample : int, optional
            Subsample factor for arrows (default: 2)
        arrow_scale : float, optional
            Arrow length scale (default: 1.0)
        arrow_color : tuple, optional
            RGB color (default: yellow)
        arrow_width : int, optional
            Arrow line width (default: 2)
        """
        self.base_preprocessor = base_preprocessor
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.subsample = subsample
        self.arrow_scale = arrow_scale
        self.arrow_color = arrow_color
        self.arrow_width = arrow_width
        
        # Detect if velocity is time-series
        self.is_timeseries = u.ndim == 3
    
    def process(self, img: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """
        Process image with velocity overlay.
        
        Parameters
        ----------
        img : np.ndarray
            Input image
        frame_idx : int, optional
            Frame index for time-series velocity
        
        Returns
        -------
        np.ndarray
            Image with velocity overlay
        """
        from .video_vectors import draw_vectors_opencv
        
        # Preprocess base image
        img_processed = self.base_preprocessor(img)
        
        # Get velocity for this frame
        if self.is_timeseries:
            idx = min(frame_idx, self.u.shape[0] - 1)
            u_frame = self.u[idx]
            v_frame = self.v[idx]
        else:
            u_frame = self.u
            v_frame = self.v
        
        # Draw vectors
        img_with_vectors = draw_vectors_opencv(
            img_processed, self.x, self.y, u_frame, v_frame,
            skip=self.subsample, scale=self.arrow_scale,
            color=self.arrow_color, thickness=self.arrow_width
        )
        
        return img_with_vectors


class GPUVelocityOverlayProcessor(VelocityOverlayProcessor):
    """
    GPU-accelerated velocity overlay processor.
    
    Stub - falls back to CPU version. Full GPU implementation
    available in complete _video package.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize GPU processor (falls back to CPU)."""
        super().__init__(*args, **kwargs)
        print("GPU overlay processor not available in stub version.")
        print("Using CPU version instead.")
    
    def process(self, img: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """Process using CPU version."""
        return super().process(img, frame_idx)
