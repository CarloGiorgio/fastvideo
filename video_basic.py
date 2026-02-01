"""
Basic Video Creation Functions
==============================

Create grayscale and color videos from microscopy stacks.
"""

import numpy as np
import cv2
import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple

from .video_writer import VideoWriter
from .video_utils import calculate_fps, normalize_image, add_text_overlay, validate_frame_range


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
                     text_position: Tuple[int, int] = (100, 100)):
    """
    Create grayscale video from microscopy stack.
    
    Parameters
    ----------
    stack : object
        Stack object with indexing and .time() method
    preprocess : callable
        Function to preprocess each frame: preprocess(img) -> normalized_img
        Output should be 2D array with values in [0, 1] or uint8
    filename : str
        Output video filename
    speed : float, optional
        Playback speed multiplier (default: 1.0)
    skip : int, optional
        Frame skip factor (default: 1)
    codec : str, optional
        Video codec: 'h264', 'h265', 'mp4v', 'xvid', 'mjpg' (default: 'h264')
    quality : str, optional
        Quality: 'high', 'medium', 'low' (default: 'medium')
    dt : float, optional
        Time spacing override (default: auto-calculate)
    fps : float, optional
        Manual FPS override (ignores speed/dt if set)
    start : int, optional
        Starting frame index (default: 0)
    end : int, optional
        Ending frame index (default: -1 = all frames)
    text : bool, optional
        Add time text overlay (default: False)
    fontsize : float, optional
        Text font size (default: 2.0)
    text_color : tuple, optional
        Text color RGB (default: white)
    text_position : tuple, optional
        Text position (x, y) (default: (100, 100))
    
    Examples
    --------
    >>> def preprocess(img):
    ...     return gaussian_filter(img, sigma=1) / img.max()
    >>> video_from_stack(stack, preprocess, 'output.mp4', speed=2, quality='high')
    """
    # Get first frame to determine dimensions
    first_frame = preprocess(stack[0].data.astype(float))
    height, width = first_frame.shape
    
    # Calculate FPS
    times = stack.time()
    if fps is None:
        fps = calculate_fps(times, speed, skip, dt)
    
    # Handle frame range
    start, end = validate_frame_range(start, end, len(stack))
    
    print(f"Creating video: {end - start} frames (skip={skip})")
    print(f"Time span: {times[start]:.2f}s - {times[end-1]:.2f}s")
    print(f"FPS: {fps:.1f}, codec: {codec}")
    
    # Create video writer
    with VideoWriter(filename, width, height, fps, 
                    is_color=False, codec=codec, quality=quality) as writer:
        
        for i in tqdm.tqdm(range(start, end, skip), desc="Rendering"):
            # Preprocess frame
            img = preprocess(stack[i].data.astype(float))
            
            # Normalize to uint8
            img = normalize_image(img)
            
            # Add text overlay if requested
            if text:
                label = f't:{times[i]:.2f}s'
                img = add_text_overlay(img, label, text_position, 
                                      fontsize, text_color)
            
            # Write frame
            writer.write(img)
    
    print('Video complete!')
    file_size_mb = Path(filename).stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")


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
                           text_position: Tuple[int, int] = (100, 100)):
    """
    Create color video from microscopy stack.
    
    Same as video_from_stack but for color/RGB output.
    Preprocess function should return (H, W, 3) array.
    
    Parameters
    ----------
    preprocess : callable
        Function to preprocess each frame: preprocess(img, frame_idx) -> rgb_img
        Output should be (H, W, 3) array with values in [0, 1] or uint8
    
    See video_from_stack for other parameters.
    
    Examples
    --------
    >>> def preprocess_color(img, idx):
    ...     # Create RGB from grayscale
    ...     norm = img / img.max()
    ...     return np.stack([norm, norm*0.5, norm*0.2], axis=-1)
    >>> video_from_stack_color(stack, preprocess_color, 'color.mp4')
    """
    # Get first frame to determine dimensions
    first_frame = preprocess(stack[0].data.astype(float), 0)
    
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError("Preprocess function must return (H, W, 3) RGB array")
    
    height, width = first_frame.shape[:2]
    
    # Calculate FPS
    times = stack.time()
    if fps is None:
        fps = calculate_fps(times, speed, skip, dt)
    
    # Handle frame range
    start, end = validate_frame_range(start, end, len(stack))
    
    print(f"Creating color video: {end - start} frames (skip={skip})")
    print(f"FPS: {fps:.1f}, codec: {codec}")
    
    # Create video writer
    with VideoWriter(filename, width, height, fps,
                    is_color=True, codec=codec, quality=quality) as writer:
        
        for i in tqdm.tqdm(range(start, end, skip), desc="Rendering"):
            # Preprocess frame (may depend on frame index)
            img = preprocess(stack[i].data.astype(float), i)
            
            # Normalize to uint8
            img = normalize_image(img)
            
            # Add text overlay if requested
            if text:
                label = f't:{times[i]:.2f}s'
                img = add_text_overlay(img, label, text_position,
                                      fontsize, text_color)
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Write frame
            writer.write(img_bgr)
    
    print('Video complete!')
    file_size_mb = Path(filename).stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")


# Backward compatibility alias
video_cv2 = video_from_stack
