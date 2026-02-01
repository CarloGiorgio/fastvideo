"""
Basic Video Creation Functions
===============================

Create grayscale and RGB videos from microscopy image stacks.
"""

import numpy as np
import cv2
import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple

from .video_writer import VideoWriter
from .video_utils import (
    calculate_fps, normalize_image, add_text_overlay, 
    validate_frame_range, print_video_info
)


def video_from_stack(stack, 
                     preprocess: Callable, 
                     filename: str,
                     speed: float = 1.0, 
                     skip: int = 1, 
                     codec: str = 'h264',
                     quality: str = 'medium', 
                     dt: Optional[float] = None,
                     fps: Optional[float] = None,
                     start: int = 0, 
                     end: int = -1,
                     text: bool = False, 
                     fontsize: float = 2.0,
                     text_color: Tuple[int, int, int] = (255, 255, 255),
                     text_position: Tuple[int, int] = (50, 50),
                     verbose: bool = True):
    """
    Create grayscale video from microscopy stack.
    
    This is the core function for creating videos from image stacks.
    It handles preprocessing, normalization, FPS calculation, and
    optional text overlays.
    
    Parameters
    ----------
    stack : object
        Stack object with indexing (.data attribute) and .time() method
    preprocess : callable
        Function to preprocess each frame: preprocess(img) -> normalized_img
        Should return 2D array with values in [0, 1] or uint8 [0, 255]
    filename : str
        Output video filename (e.g., 'output.mp4')
    speed : float, optional
        Playback speed multiplier (2.0 = 2x speed), default 1.0
    skip : int, optional
        Frame skip factor (2 = use every 2nd frame), default 1
    codec : str, optional
        Video codec: 'h264', 'h265', 'mp4v', 'xvid', 'mjpg', default 'h264'
    quality : str, optional
        Quality preset: 'high', 'medium', 'low', default 'medium'
    dt : float, optional
        Time spacing override. If None, auto-calculated from stack.time()
    fps : float, optional
        Manual FPS override. If set, ignores speed/dt calculation
    start : int, optional
        Starting frame index, default 0
    end : int, optional
        Ending frame index. -1 = use all frames, default -1
    text : bool, optional
        Add time text overlay, default False
    fontsize : float, optional
        Text font size scale, default 2.0
    text_color : tuple of int, optional
        Text color (R, G, B), default white (255, 255, 255)
    text_position : tuple of int, optional
        Text position (x, y) pixels, default (50, 50)
    verbose : bool, optional
        Print progress information, default True
    
    Returns
    -------
    None
        Video file is written to disk
    
    Examples
    --------
    >>> # Basic usage with normalization
    >>> def preprocess(img):
    ...     return (img - img.min()) / (img.max() - img.min())
    >>> 
    >>> video_from_stack(stack, preprocess, 'output.mp4')
    
    >>> # High quality with 2x speed
    >>> video_from_stack(stack, preprocess, 'fast.mp4', 
    ...                  speed=2.0, quality='high')
    
    >>> # With time overlay
    >>> video_from_stack(stack, preprocess, 'time.mp4',
    ...                  text=True, fontsize=1.5)
    
    >>> # Process subset of frames
    >>> video_from_stack(stack, preprocess, 'subset.mp4',
    ...                  start=100, end=500, skip=2)
    
    Notes
    -----
    - Preprocessing function should handle normalization appropriately
    - For best compression, use H264 or H265 codecs
    - Skip parameter is useful for large datasets to reduce file size
    - Text overlay uses time from stack.time() method
    """
    # Get first frame to determine dimensions
    first_frame = preprocess(stack[0].data.astype(float))
    if first_frame.ndim != 2:
        raise ValueError(
            f"Preprocess must return 2D array for grayscale video, "
            f"got shape {first_frame.shape}"
        )
    height, width = first_frame.shape
    
    # Get timing information
    times = stack.time()
    
    # Calculate FPS
    if fps is None:
        fps = calculate_fps(times, speed, skip, dt)
    
    # Handle frame range
    start, end = validate_frame_range(start, end, len(stack))
    n_frames = (end - start) // skip
    
    # Print info
    if verbose:
        print_video_info(width, height, fps, n_frames, codec, quality,
                        times[start], times[min(end-1, len(times)-1)])
    
    # Create video writer
    with VideoWriter(filename, width, height, fps, 
                    is_color=False, codec=codec, quality=quality) as writer:
        
        # Process frames with progress bar
        iterator = range(start, end, skip)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc="Rendering video")
        
        for i in iterator:
            # Preprocess frame
            img = preprocess(stack[i].data.astype(float))
            
            # Normalize to uint8
            img = normalize_image(img, output_range='uint8')
            
            # Add text overlay if requested
            if text:
                label = f't: {times[i]:.2f}s'
                img = add_text_overlay(img, label, text_position, 
                                      fontsize, text_color)
            
            # Write frame
            writer.write(img)
    
    # Report completion
    if verbose:
        file_size_mb = Path(filename).stat().st_size / (1024**2)
        print(f'✓ Video complete: {filename}')
        print(f'  File size: {file_size_mb:.1f} MB')


def video_from_stack_color(stack, 
                           preprocess: Callable, 
                           filename: str,
                           speed: float = 1.0, 
                           skip: int = 1, 
                           codec: str = 'h264',
                           quality: str = 'medium', 
                           dt: Optional[float] = None,
                           fps: Optional[float] = None,
                           start: int = 0, 
                           end: int = -1,
                           text: bool = False, 
                           fontsize: float = 2.0,
                           text_color: Tuple[int, int, int] = (255, 255, 255),
                           text_position: Tuple[int, int] = (50, 50),
                           verbose: bool = True):
    """
    Create RGB color video from microscopy stack.
    
    Similar to video_from_stack but expects preprocessing function
    to return RGB images with shape (height, width, 3).
    
    Parameters
    ----------
    stack : object
        Stack object with indexing and .time() method
    preprocess : callable
        Function: preprocess(img) -> rgb_img
        Must return shape (height, width, 3) in RGB format
    filename : str
        Output video filename
    speed : float, optional
        Playback speed multiplier, default 1.0
    skip : int, optional
        Frame skip factor, default 1
    codec : str, optional
        Video codec, default 'h264'
    quality : str, optional
        Quality preset, default 'medium'
    dt : float, optional
        Time spacing override
    fps : float, optional
        Manual FPS override
    start : int, optional
        Starting frame index, default 0
    end : int, optional
        Ending frame index, default -1
    text : bool, optional
        Add text overlay, default False
    fontsize : float, optional
        Text size, default 2.0
    text_color : tuple, optional
        Text color RGB, default white
    text_position : tuple, optional
        Text position, default (50, 50)
    verbose : bool, optional
        Print progress, default True
    
    Examples
    --------
    >>> # RGB colormap
    >>> import matplotlib.pyplot as plt
    >>> def preprocess_rgb(img):
    ...     img_norm = (img - img.min()) / (img.max() - img.min())
    ...     rgb = plt.cm.viridis(img_norm)[:, :, :3]  # Drop alpha
    ...     return (rgb * 255).astype(np.uint8)
    >>> 
    >>> video_from_stack_color(stack, preprocess_rgb, 'color.mp4')
    """
    # Get first frame to determine dimensions
    first_frame = preprocess(stack[0].data.astype(float))
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(
            f"Preprocess must return (height, width, 3) for color video, "
            f"got shape {first_frame.shape}"
        )
    height, width = first_frame.shape[:2]
    
    # Get timing information
    times = stack.time()
    
    # Calculate FPS
    if fps is None:
        fps = calculate_fps(times, speed, skip, dt)
    
    # Handle frame range
    start, end = validate_frame_range(start, end, len(stack))
    n_frames = (end - start) // skip
    
    # Print info
    if verbose:
        print_video_info(width, height, fps, n_frames, codec, quality,
                        times[start], times[min(end-1, len(times)-1)])
    
    # Create video writer
    with VideoWriter(filename, width, height, fps, 
                    is_color=True, codec=codec, quality=quality) as writer:
        
        # Process frames
        iterator = range(start, end, skip)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc="Rendering video")
        
        for i in iterator:
            # Preprocess frame - should return RGB
            img = preprocess(stack[i].data.astype(float))
            
            # Ensure uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Add text overlay if requested
            if text:
                label = f't: {times[i]:.2f}s'
                img = add_text_overlay(img, label, text_position, 
                                      fontsize, text_color)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write frame
            writer.write(img_bgr)
    
    # Report completion
    if verbose:
        file_size_mb = Path(filename).stat().st_size / (1024**2)
        print(f'✓ Video complete: {filename}')
        print(f'  File size: {file_size_mb:.1f} MB')


# Backward compatibility alias
video_cv2 = video_from_stack
