"""
Additional Utilities for Video Creation
=======================================

Helper functions for resolution optimization and parameter calibration.
"""

import numpy as np
from typing import Tuple, Optional


def mm2inch(mm: float) -> float:
    """
    Convert millimeters to inches.
    
    Parameters
    ----------
    mm : float
        Measurement in millimeters
    
    Returns
    -------
    float
        Measurement in inches
    """
    return mm / 25.4


def calculate_optimal_resolution(original_size: Tuple[int, int],
                                 target_size_mb: float,
                                 duration_s: float,
                                 fps: float,
                                 codec: str = 'h264') -> Tuple[int, int]:
    """
    Calculate optimal resolution to meet target file size.
    
    Uses empirical bitrate estimates for different codecs.
    
    Parameters
    ----------
    original_size : tuple
        Original (width, height)
    target_size_mb : float
        Target file size in MB
    duration_s : float
        Video duration in seconds
    fps : float
        Frames per second
    codec : str, optional
        Video codec (default: 'h264')
    
    Returns
    -------
    tuple
        Optimized (width, height)
    
    Examples
    --------
    >>> optimal_size = calculate_optimal_resolution(
    ...     (1920, 1080), target_size_mb=100, duration_s=60, fps=30
    ... )
    """
    from .video_utils import scale_resolution
    
    return scale_resolution(original_size, target_size_mb, fps, duration_s, codec)


def auto_calibrate_arrows(u: np.ndarray, v: np.ndarray,
                          pixel_size: float,
                          target_length_pixels: float = 20.0,
                          subsample: int = 2) -> dict:
    """
    Automatically calibrate arrow visualization parameters.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity components
    pixel_size : float
        Physical size per pixel (e.g., microns/pixel)
    target_length_pixels : float, optional
        Target arrow length in pixels (default: 20)
    subsample : int, optional
        Recommended subsample factor (default: 2)
    
    Returns
    -------
    dict
        Calibrated parameters: 'scale', 'subsample', 'thickness'
    
    Examples
    --------
    >>> params = auto_calibrate_arrows(u, v, pixel_size=0.65)
    >>> video_with_vectors(stack, preprocess, x, y, u, v, 'out.mp4',
    ...                   vector_scale=params['scale'],
    ...                   vector_skip=params['subsample'])
    """
    from .video_vectors import calculate_auto_arrow_scale
    
    # Calculate scale
    scale = calculate_auto_arrow_scale(u, v, pixel_size, target_length_pixels)
    
    # Calculate velocity statistics
    velocity_mag = np.sqrt(u**2 + v**2)
    median_vel = np.median(velocity_mag[velocity_mag > 0])
    max_vel = np.percentile(velocity_mag[velocity_mag > 0], 95)
    
    # Adjust subsample based on field density
    grid_size = min(u.shape[-2:])
    if grid_size > 512:
        subsample = 4
    elif grid_size > 256:
        subsample = 3
    else:
        subsample = 2
    
    # Adjust thickness based on velocity range
    if max_vel / median_vel > 5:
        thickness = 3  # High variability, thicker arrows
    else:
        thickness = 2  # Normal thickness
    
    params = {
        'scale': scale,
        'subsample': subsample,
        'thickness': thickness,
        'median_velocity': median_vel,
        'max_velocity': max_vel
    }
    
    print(f"Auto-calibrated arrow parameters:")
    print(f"  Scale: {scale:.2f}")
    print(f"  Subsample: {subsample}")
    print(f"  Thickness: {thickness}")
    print(f"  Median velocity: {median_vel:.2f}")
    print(f"  95th percentile velocity: {max_vel:.2f}")
    
    return params


def estimate_file_size(width: int, height: int, 
                      duration_s: float, fps: float,
                      codec: str = 'h264', quality: str = 'medium') -> float:
    """
    Estimate output file size.
    
    Parameters
    ----------
    width, height : int
        Video dimensions
    duration_s : float
        Duration in seconds
    fps : float
        Frames per second
    codec : str, optional
        Video codec (default: 'h264')
    quality : str, optional
        Quality setting (default: 'medium')
    
    Returns
    -------
    float
        Estimated file size in MB
    
    Examples
    --------
    >>> size_mb = estimate_file_size(1920, 1080, 60, 30)
    >>> print(f"Estimated size: {size_mb:.1f} MB")
    """
    # Bitrate estimates (Mbps per megapixel)
    BITRATE_PER_MEGAPIXEL = {
        'h264': {'high': 3.0, 'medium': 2.0, 'low': 1.0},
        'h265': {'high': 1.5, 'medium': 1.0, 'low': 0.5},
        'mp4v': {'high': 4.0, 'medium': 3.0, 'low': 2.0},
    }
    
    # Get bitrate factor
    codec_rates = BITRATE_PER_MEGAPIXEL.get(codec, {'medium': 2.0})
    bitrate_factor = codec_rates.get(quality, 2.0)
    
    # Calculate file size
    megapixels = (width * height) / 1e6
    bitrate_mbps = megapixels * bitrate_factor
    file_size_mb = (bitrate_mbps * duration_s) / 8
    
    return file_size_mb


def validate_inputs(stack, x: np.ndarray, y: np.ndarray,
                   u: np.ndarray, v: np.ndarray) -> bool:
    """
    Validate inputs for video creation.
    
    Parameters
    ----------
    stack : object
        Stack object
    x, y : np.ndarray
        Coordinate arrays
    u, v : np.ndarray
        Velocity components
    
    Returns
    -------
    bool
        True if inputs are valid
    
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    # Check stack
    if not hasattr(stack, '__len__'):
        raise ValueError("Stack must be indexable")
    
    if len(stack) == 0:
        raise ValueError("Stack is empty")
    
    # Check coordinate arrays
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes don't match: {x.shape} vs {y.shape}")
    
    # Check velocity arrays
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes don't match: {u.shape} vs {v.shape}")
    
    # Check velocity vs coordinates
    if u.ndim == 3:
        # Time-series velocity
        if u.shape[1:] != x.shape:
            raise ValueError(
                f"Velocity spatial shape {u.shape[1:]} doesn't match "
                f"coordinate shape {x.shape}"
            )
        
        if u.shape[0] != len(stack):
            print(f"Warning: Velocity frames ({u.shape[0]}) != "
                  f"stack frames ({len(stack)})")
    else:
        # Single velocity field
        if u.shape != x.shape:
            raise ValueError(
                f"Velocity shape {u.shape} doesn't match "
                f"coordinate shape {x.shape}"
            )
    
    print("✓ Inputs validated")
    return True
