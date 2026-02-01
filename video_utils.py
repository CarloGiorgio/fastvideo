"""
Video Utility Functions
=======================

Helper functions for video creation, normalization, and text overlays.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def calculate_fps(times: np.ndarray, speed: float = 1.0, 
                 skip: int = 1, dt: Optional[float] = None) -> float:
    """
    Calculate frames per second for video creation.
    
    Parameters
    ----------
    times : np.ndarray
        Array of timestamps
    speed : float, optional
        Playback speed multiplier (default: 1.0)
    skip : int, optional
        Frame skip factor (default: 1)
    dt : float, optional
        Manual time spacing override (default: auto-calculate)
    
    Returns
    -------
    float
        Calculated FPS, minimum 1.0
    
    Examples
    --------
    >>> times = np.linspace(0, 100, 1000)  # 100 seconds, 1000 frames
    >>> fps = calculate_fps(times, speed=2.0)  # 2x speed
    >>> print(f"FPS: {fps:.1f}")
    """
    if dt is None:
        # Calculate from time array
        if len(times) > 1:
            dt = np.mean(np.diff(times))
        else:
            dt = 1.0  # Default if only one frame
    
    # FPS = speed / (dt * skip)
    fps = speed / (dt * skip)
    
    # Ensure minimum FPS
    fps = max(1.0, fps)
    
    return fps


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 range [0, 255].
    
    Handles both float [0, 1] and arbitrary range arrays.
    Preserves RGB structure if present.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (float or uint8)
    
    Returns
    -------
    np.ndarray
        Normalized uint8 image
    
    Examples
    --------
    >>> img_float = np.random.rand(100, 100)
    >>> img_uint8 = normalize_image(img_float)
    >>> print(img_uint8.dtype, img_uint8.min(), img_uint8.max())
    """
    if img.dtype == np.uint8:
        return img
    
    # Normalize to [0, 1]
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img)
    
    # Convert to uint8
    return (img_norm * 255).astype(np.uint8)


def add_text_overlay(img: np.ndarray, text: str, 
                     position: Tuple[int, int] = (50, 50),
                     fontsize: float = 1.5,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2) -> np.ndarray:
    """
    Add text overlay to image.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (uint8)
    text : str
        Text to overlay
    position : tuple, optional
        (x, y) position (default: (50, 50))
    fontsize : float, optional
        Font size (default: 1.5)
    color : tuple, optional
        RGB color (default: white)
    thickness : int, optional
        Text thickness (default: 2)
    
    Returns
    -------
    np.ndarray
        Image with text overlay
    
    Examples
    --------
    >>> img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> img_with_text = add_text_overlay(img, "t=1.5s", (10, 30))
    """
    # Make a copy to avoid modifying original
    img_copy = img.copy()
    
    # Convert grayscale to RGB if needed
    if img_copy.ndim == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2RGB)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_copy, text, position, font, fontsize, color, thickness)
    
    return img_copy


def get_colormap_rgb(value: float, vmin: float = 0.0, vmax: float = 1.0,
                     cmap_name: str = 'viridis') -> Tuple[int, int, int]:
    """
    Get RGB color from matplotlib colormap.
    
    Parameters
    ----------
    value : float
        Value to map
    vmin, vmax : float, optional
        Value range (default: 0 to 1)
    cmap_name : str, optional
        Matplotlib colormap name (default: 'viridis')
    
    Returns
    -------
    tuple
        RGB color (0-255)
    
    Examples
    --------
    >>> color = get_colormap_rgb(0.5, 0, 1, 'jet')
    >>> print(color)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Normalize value
        norm_value = (value - vmin) / (vmax - vmin)
        norm_value = np.clip(norm_value, 0, 1)
        
        # Get colormap
        cmap = plt.get_cmap(cmap_name)
        rgba = cmap(norm_value)
        
        # Convert to RGB (0-255)
        rgb = tuple(int(c * 255) for c in rgba[:3])
        return rgb
        
    except ImportError:
        # Fallback if matplotlib not available
        return (255, 255, 255)


def scale_resolution(original_size: Tuple[int, int], 
                     target_size_mb: float,
                     fps: float,
                     duration_s: float,
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
    fps : float
        Frames per second
    duration_s : float
        Video duration in seconds
    codec : str, optional
        Video codec (default: 'h264')
    
    Returns
    -------
    tuple
        Scaled (width, height)
    
    Examples
    --------
    >>> new_size = scale_resolution((1920, 1080), 50.0, 30.0, 10.0)
    >>> print(f"Scaled to: {new_size}")
    """
    # Bitrate estimates (Mbps per megapixel)
    BITRATE_PER_MEGAPIXEL = {
        'h264': 2.0,
        'h265': 1.0,
        'mp4v': 3.0,
        'xvid': 2.5,
        'mjpg': 5.0
    }
    
    bitrate_factor = BITRATE_PER_MEGAPIXEL.get(codec, 2.0)
    
    # Calculate current file size estimate
    width, height = original_size
    megapixels = (width * height) / 1e6
    bitrate_mbps = megapixels * bitrate_factor
    file_size_mb = (bitrate_mbps * duration_s) / 8
    
    # Calculate scaling factor
    if file_size_mb > target_size_mb:
        scale_factor = np.sqrt(target_size_mb / file_size_mb)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure even dimensions (required by some codecs)
        new_width = (new_width // 2) * 2
        new_height = (new_height // 2) * 2
        
        return (new_width, new_height)
    
    return original_size


def validate_frame_range(start: int, end: int, stack_length: int) -> Tuple[int, int]:
    """
    Validate and fix frame range parameters.
    
    Parameters
    ----------
    start : int
        Starting frame index
    end : int
        Ending frame index (negative means from end)
    stack_length : int
        Total number of frames
    
    Returns
    -------
    tuple
        Validated (start, end) indices
    
    Examples
    --------
    >>> start, end = validate_frame_range(0, -1, 1000)
    >>> print(f"Processing frames {start} to {end}")
    """
    # Handle negative end index
    if end <= 0:
        end = max(start + 1, stack_length + end)
    else:
        end = max(start + 1, min(end, stack_length))
    
    # Ensure valid range
    start = max(0, min(start, stack_length - 1))
    end = max(start + 1, min(end, stack_length))
    
    return start, end
