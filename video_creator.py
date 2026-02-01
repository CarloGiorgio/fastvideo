"""
High-Level Video Creation Functions
====================================

Convenience wrappers for common video creation workflows.
"""

from typing import Callable, Optional
import numpy as np

from .video_basic import video_from_stack
from .video_vectors import video_with_vectors, calculate_auto_arrow_scale


def create_video_simple(stack, preprocess: Callable, filename: str, **kwargs):
    """
    Simple video creation with sensible defaults.
    
    Parameters
    ----------
    stack : object
        Stack with .data and .time() methods
    preprocess : callable
        Preprocessing function
    filename : str
        Output filename
    **kwargs
        Additional parameters passed to video_from_stack
    
    Examples
    --------
    >>> create_video_simple(stack, lambda x: x/x.max(), 'output.mp4')
    """
    kwargs.setdefault('codec', 'h264')
    kwargs.setdefault('quality', 'medium')
    kwargs.setdefault('speed', 1.0)
    
    return video_from_stack(stack, preprocess, filename, **kwargs)


def create_velocity_video(stack, preprocess: Callable, 
                         x: np.ndarray, y: np.ndarray,
                         u: np.ndarray, v: np.ndarray,
                         filename: str,
                         pixel_size: float,
                         **kwargs):
    """
    Create velocity field video with automatic parameter calibration.
    
    Parameters
    ----------
    stack : object
        Stack with .data and .time() methods
    preprocess : callable
        Preprocessing function
    x, y : np.ndarray
        Coordinate meshgrids
    u, v : np.ndarray
        Velocity components
    filename : str
        Output filename
    pixel_size : float
        Physical size per pixel (for arrow scaling)
    **kwargs
        Additional parameters
    
    Examples
    --------
    >>> create_velocity_video(stack, preprocess, x, y, u, v,
    ...                      'velocity.mp4', pixel_size=0.65)
    """
    # Auto-calculate arrow scale if not provided
    if 'vector_scale' not in kwargs:
        kwargs['vector_scale'] = calculate_auto_arrow_scale(u, v, pixel_size)
        print(f"Auto-calibrated arrow scale: {kwargs['vector_scale']:.2f}")
    
    # Set defaults
    kwargs.setdefault('codec', 'h264')
    kwargs.setdefault('quality', 'medium')
    kwargs.setdefault('vector_skip', 2)
    kwargs.setdefault('vector_color', (255, 255, 0))
    
    return video_with_vectors(stack, preprocess, x, y, u, v, filename, **kwargs)


def create_video_with_validation(stack, preprocess: Callable, filename: str,
                                 preview: bool = True, **kwargs):
    """
    Create video with optional preview validation.
    
    Parameters
    ----------
    stack : object
        Stack
    preprocess : callable
        Preprocessing function
    filename : str
        Output filename
    preview : bool, optional
        Show preview before creating video
    **kwargs
        Additional parameters
    
    Examples
    --------
    >>> create_video_with_validation(stack, preprocess, 'output.mp4', preview=True)
    """
    if preview:
        print("Preview functionality requires full _video package")
        print("Creating video without preview...")
    
    return create_video_simple(stack, preprocess, filename, **kwargs)
