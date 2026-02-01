"""
Video Utilities
===============

Utility functions for video creation: FPS calculation, image normalization,
text overlays, and parameter validation.
"""

import numpy as np
import cv2
from typing import Callable, Optional, Tuple


def calculate_fps(times: np.ndarray, speed: float = 1.0, skip: int = 1, 
                  dt: Optional[float] = None) -> float:
    """
    Calculate frames per second from timing information.
    
    Parameters
    ----------
    times : np.ndarray
        Array of timestamps for each frame
    speed : float, optional
        Playback speed multiplier (default: 1.0)
    skip : int, optional
        Frame skip factor (default: 1)
    dt : float, optional
        Override time spacing. If None, calculated from times array.
    
    Returns
    -------
    float
        Frames per second (minimum 1.0)
    
    Examples
    --------
    >>> times = np.linspace(0, 10, 300)  # 300 frames over 10 seconds
    >>> fps = calculate_fps(times, speed=2.0)  # 2x playback speed
    >>> print(f"FPS: {fps:.1f}")
    FPS: 60.0
    """
    if dt is None:
        # Calculate average time spacing
        dt = np.mean(np.diff(times[1:]))
    
    # FPS = speed / (dt * skip)
    fps = speed / (dt * skip)
    
    # Ensure minimum FPS of 1
    fps = max(1.0, fps)
    
    return fps


def normalize_image(img: np.ndarray, output_range: str = 'uint8') -> np.ndarray:
    """
    Normalize image to specified output range.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (any dtype, any range)
    output_range : str, optional
        Output format: 'uint8' (0-255) or 'float' (0-1), default 'uint8'
    
    Returns
    -------
    np.ndarray
        Normalized image
    
    Examples
    --------
    >>> img = np.random.rand(100, 100) * 1000  # Random image
    >>> img_norm = normalize_image(img)
    >>> print(img_norm.dtype, img_norm.min(), img_norm.max())
    uint8 0 255
    """
    # Handle different input types
    if img.dtype == np.uint8 and output_range == 'uint8':
        return img
    
    # Convert to float for processing
    img_float = img.astype(np.float64)
    
    # Normalize to [0, 1]
    img_min = img_float.min()
    img_max = img_float.max()
    
    if img_max - img_min > 0:
        img_norm = (img_float - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(img_float)
    
    # Convert to desired output
    if output_range == 'uint8':
        return (img_norm * 255).astype(np.uint8)
    else:
        return img_norm.astype(np.float32)


def add_text_overlay(img: np.ndarray, text: str, 
                     position: Tuple[int, int] = (100, 100),
                     fontsize: float = 2.0,
                     color: Tuple[int, int, int] = (255, 255, 255),
                     thickness: int = 2,
                     font: int = cv2.FONT_HERSHEY_SIMPLEX,
                     background: bool = True,
                     background_color: Tuple[int, int, int] = (0, 0, 0),
                     background_alpha: float = 0.7) -> np.ndarray:
    """
    Add text overlay to image with optional background.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale or RGB)
    text : str
        Text to overlay
    position : tuple of int, optional
        Text position (x, y), default (100, 100)
    fontsize : float, optional
        Font scale, default 2.0
    color : tuple of int, optional
        Text color (R, G, B), default white
    thickness : int, optional
        Text thickness, default 2
    font : int, optional
        OpenCV font type, default FONT_HERSHEY_SIMPLEX
    background : bool, optional
        Add semi-transparent background, default True
    background_color : tuple of int, optional
        Background color (R, G, B), default black
    background_alpha : float, optional
        Background transparency (0=transparent, 1=opaque), default 0.7
    
    Returns
    -------
    np.ndarray
        Image with text overlay
    
    Examples
    --------
    >>> img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    >>> img_text = add_text_overlay(img, "Frame 42", (50, 50))
    """
    # Ensure img is uint8
    if img.dtype != np.uint8:
        img = normalize_image(img, output_range='uint8')
    
    # Convert grayscale to RGB if needed
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Make a copy to avoid modifying original
    img_out = img.copy()
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, fontsize, thickness
    )
    
    # Add background if requested
    if background:
        # Calculate background rectangle
        x, y = position
        padding = 10
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - text_height - padding)
        bg_x2 = min(img_out.shape[1], x + text_width + padding)
        bg_y2 = min(img_out.shape[0], y + padding)
        
        # Create background overlay
        overlay = img_out.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                     background_color, -1)
        
        # Blend with original
        cv2.addWeighted(overlay, background_alpha, img_out, 1 - background_alpha, 
                       0, img_out)
    
    # Add text
    cv2.putText(img_out, text, position, font, fontsize, color, 
               thickness, cv2.LINE_AA)
    
    return img_out


def validate_frame_range(start: int, end: int, stack_length: int) -> Tuple[int, int]:
    """
    Validate and adjust frame range indices.
    
    Parameters
    ----------
    start : int
        Starting frame index
    end : int
        Ending frame index (-1 means last frame)
    stack_length : int
        Total number of frames in stack
    
    Returns
    -------
    tuple of int
        Validated (start, end) indices
    
    Raises
    ------
    ValueError
        If frame range is invalid
    
    Examples
    --------
    >>> start, end = validate_frame_range(0, -1, 1000)
    >>> print(start, end)
    0 1000
    
    >>> start, end = validate_frame_range(100, 200, 1000)
    >>> print(start, end)
    100 200
    """
    # Handle negative indices
    if start < 0:
        start = 0
    
    if end < 0 or end > stack_length:
        end = stack_length
    
    # Validate range
    if start >= end:
        raise ValueError(
            f"Invalid frame range: start={start}, end={end}. "
            f"Start must be less than end."
        )
    
    if start >= stack_length:
        raise ValueError(
            f"Start frame {start} exceeds stack length {stack_length}"
        )
    
    return start, end


def calculate_output_dimensions(input_width: int, input_height: int,
                                max_width: Optional[int] = None,
                                max_height: Optional[int] = None,
                                scale: Optional[float] = None) -> Tuple[int, int]:
    """
    Calculate output video dimensions with constraints.
    
    Ensures dimensions are even (required by most codecs) and respects
    maximum size constraints while maintaining aspect ratio.
    
    Parameters
    ----------
    input_width : int
        Original width
    input_height : int
        Original height
    max_width : int, optional
        Maximum allowed width
    max_height : int, optional
        Maximum allowed height
    scale : float, optional
        Scaling factor (e.g., 0.5 for half size)
    
    Returns
    -------
    tuple of int
        (width, height) as even integers
    
    Examples
    --------
    >>> w, h = calculate_output_dimensions(1920, 1080, max_width=1280)
    >>> print(w, h)
    1280 720
    
    >>> w, h = calculate_output_dimensions(1920, 1080, scale=0.5)
    >>> print(w, h)
    960 540
    """
    width = input_width
    height = input_height
    
    # Apply scale if provided
    if scale is not None:
        width = int(width * scale)
        height = int(height * scale)
    
    # Apply max constraints
    if max_width is not None and width > max_width:
        aspect_ratio = height / width
        width = max_width
        height = int(width * aspect_ratio)
    
    if max_height is not None and height > max_height:
        aspect_ratio = width / height
        height = max_height
        width = int(height * aspect_ratio)
    
    # Ensure even dimensions
    width = width + (width % 2)
    height = height + (height % 2)
    
    return width, height


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable time string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
    
    Returns
    -------
    str
        Formatted time string
    
    Examples
    --------
    >>> print(format_time(65.5))
    1m 5.5s
    
    >>> print(format_time(3665))
    1h 1m 5s
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def estimate_file_size(width: int, height: int, n_frames: int, fps: float,
                      codec: str = 'h264', quality: str = 'medium') -> float:
    """
    Estimate output video file size in MB.
    
    This is a rough estimate based on typical bitrates for different
    codecs and quality settings.
    
    Parameters
    ----------
    width : int
        Video width
    height : int
        Video height
    n_frames : int
        Number of frames
    fps : float
        Frames per second
    codec : str, optional
        Video codec, default 'h264'
    quality : str, optional
        Quality setting, default 'medium'
    
    Returns
    -------
    float
        Estimated file size in MB
    
    Examples
    --------
    >>> size_mb = estimate_file_size(1920, 1080, 1000, 30, 'h264', 'high')
    >>> print(f"Estimated size: {size_mb:.1f} MB")
    Estimated size: 45.7 MB
    """
    # Typical bitrates in kbps for 1080p
    bitrates = {
        ('h264', 'high'): 8000,
        ('h264', 'medium'): 5000,
        ('h264', 'low'): 2500,
        ('h265', 'high'): 5000,
        ('h265', 'medium'): 3000,
        ('h265', 'low'): 1500,
        ('mp4v', 'high'): 10000,
        ('mp4v', 'medium'): 6000,
        ('mp4v', 'low'): 3000,
    }
    
    # Get base bitrate (default to h264 medium if not found)
    base_bitrate = bitrates.get((codec, quality), 5000)
    
    # Scale by resolution (relative to 1080p)
    pixels = width * height
    reference_pixels = 1920 * 1080
    scale_factor = pixels / reference_pixels
    
    bitrate_kbps = base_bitrate * scale_factor
    
    # Calculate duration and file size
    duration_sec = n_frames / fps
    size_mb = (bitrate_kbps * duration_sec) / 8 / 1024
    
    return size_mb


def print_video_info(width: int, height: int, fps: float, n_frames: int,
                    codec: str, quality: str, start_time: float = 0,
                    end_time: Optional[float] = None):
    """
    Print formatted video information.
    
    Parameters
    ----------
    width : int
        Video width
    height : int
        Video height
    fps : float
        Frames per second
    n_frames : int
        Number of frames
    codec : str
        Video codec
    quality : str
        Quality setting
    start_time : float, optional
        Start time in seconds, default 0
    end_time : float, optional
        End time in seconds
    """
    duration = n_frames / fps
    if end_time is None:
        end_time = start_time + duration * n_frames
    
    print("\n" + "="*60)
    print("VIDEO CREATION INFO")
    print("="*60)
    print(f"Resolution:  {width} × {height}")
    print(f"Frames:      {n_frames}")
    print(f"FPS:         {fps:.1f}")
    print(f"Duration:    {format_time(duration)}")
    print(f"Time span:   {start_time:.2f}s - {end_time:.2f}s")
    print(f"Codec:       {codec}")
    print(f"Quality:     {quality}")
    
    est_size = estimate_file_size(width, height, n_frames, fps, codec, quality)
    print(f"Est. size:   {est_size:.1f} MB")
    print("="*60 + "\n")
