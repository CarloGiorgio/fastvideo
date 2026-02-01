"""
Additional Utilities
====================

Helper functions for resolution optimization and arrow calibration.
"""

import numpy as np
from typing import Optional, Tuple

def calculate_optimal_resolution(original_width: int, original_height: int,
                                 max_resolution: Tuple[int, int] = (1920, 1080),
                                 target_size_mb: float = 200.0) -> Tuple[int, int]:
    """
    Calculate optimal output resolution for target file size.
    
    Parameters
    ----------
    original_width, original_height : int
        Original dimensions
    max_resolution : tuple, optional
        Maximum allowed resolution
    target_size_mb : float, optional
        Target file size in MB
    
    Returns
    -------
    tuple
        (width, height) optimized dimensions
    """
    # Simple downscaling logic
    scale = 1.0
    if original_width > max_resolution[0]:
        scale = min(scale, max_resolution[0] / original_width)
    if original_height > max_resolution[1]:
        scale = min(scale, max_resolution[1] / original_height)
    
    # Apply scale
    width = int(original_width * scale)
    height = int(original_height * scale)
    
    # Ensure even dimensions
    width += width % 2
    height += height % 2
    
    return width, height

def auto_calibrate_arrows(u: np.ndarray, v: np.ndarray, 
                          pixel_size: float) -> dict:
    """
    Auto-calibrate arrow parameters for visualization.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity fields
    pixel_size : float
        Pixel size in physical units
    
    Returns
    -------
    dict
        Dictionary with recommended parameters
    """
    from .video_vectors import calculate_auto_arrow_scale
    
    scale = calculate_auto_arrow_scale(u, v, pixel_size)
    
    # Calculate recommended subsample based on field density
    if u.shape[0] > 100:
        subsample = 3
    elif u.shape[0] > 50:
        subsample = 2
    else:
        subsample = 1
    
    return {
        'vector_scale': scale,
        'vector_skip': subsample,
        'vector_thickness': 2,
        'vector_color': (255, 255, 0)
    }

def mm2inch(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm / 25.4
