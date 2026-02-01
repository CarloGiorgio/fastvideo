"""
Vector Field Video Creation
============================

Functions for rendering velocity fields and vector overlays on videos.
Optimized for PIV (Particle Image Velocimetry) visualization.
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

# Optional GPU support
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def draw_vectors_opencv(img: np.ndarray, x: np.ndarray, y: np.ndarray,
                        u: np.ndarray, v: np.ndarray,
                        skip: int = 2, scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 0),
                        thickness: int = 2, arrow_tip: float = 0.3) -> np.ndarray:
    """
    Draw vector field on image using OpenCV (fast implementation).
    
    This function overlays velocity vectors as arrows on an image.
    Much faster than matplotlib-based approaches.
    
    Parameters
    ----------
    img : np.ndarray
        Background image (grayscale or RGB). Will be converted to RGB if grayscale.
    x, y : np.ndarray
        Meshgrid coordinates of vector field (2D arrays)
    u, v : np.ndarray
        Vector components (same shape as x, y)
    skip : int, optional
        Draw every N-th vector for performance (default: 2)
    scale : float, optional
        Vector length scaling factor (default: 1.0)
    color : tuple of int, optional
        Arrow color in RGB format (default: yellow (255, 255, 0))
    thickness : int, optional
        Arrow line thickness in pixels (default: 2)
    arrow_tip : float, optional
        Arrow tip size ratio (default: 0.3)
    
    Returns
    -------
    np.ndarray
        Image with vectors overlaid (RGB uint8)
    
    Examples
    --------
    >>> # Create simple vector field
    >>> img = np.random.rand(512, 512)
    >>> x, y = np.meshgrid(np.arange(512), np.arange(512))
    >>> u = np.ones_like(x) * 5
    >>> v = np.ones_like(y) * 3
    >>> img_arrows = draw_vectors_opencv(img, x, y, u, v, skip=16)
    
    >>> # PIV visualization with custom colors
    >>> img_piv = draw_vectors_opencv(
    ...     img, x_piv, y_piv, u_piv, v_piv,
    ...     skip=3, scale=2.0, color=(0, 255, 0), thickness=3
    ... )
    
    Notes
    -----
    - Vectors with magnitude < 0.5 pixels are automatically skipped
    - Vectors outside image bounds are skipped
    - For large datasets, increase `skip` parameter for better performance
    """
    # Convert to RGB if grayscale
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()
    
    # Ensure uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = normalize_image(img_rgb, output_range='uint8')
    
    # Draw arrows (subsample for performance)
    for i in range(0, x.shape[0], skip):
        for j in range(0, x.shape[1], skip):
            x0, y0 = int(x[i, j]), int(y[i, j])
            dx, dy = u[i, j] * scale, v[i, j] * scale
            x1, y1 = int(x0 + dx), int(y0 + dy)
            
            # Skip very small vectors (noise)
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            
            # Skip if start point is out of bounds
            if not (0 <= x0 < img_rgb.shape[1] and 0 <= y0 < img_rgb.shape[0]):
                continue
            
            # Clip end point to image bounds
            x1 = np.clip(x1, 0, img_rgb.shape[1] - 1)
            y1 = np.clip(y1, 0, img_rgb.shape[0] - 1)
            
            # Draw arrow
            cv2.arrowedLine(img_rgb, (x0, y0), (x1, y1), 
                          color, thickness, tipLength=arrow_tip)
    
    return img_rgb


def draw_vectors_gpu(img: np.ndarray, x: np.ndarray, y: np.ndarray,
                     u: np.ndarray, v: np.ndarray,
                     skip: int = 2, scale: float = 1.0,
                     color: Tuple[int, int, int] = (255, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    GPU-accelerated vector drawing (requires CuPy).
    
    Currently falls back to CPU version. Full GPU implementation would
    require custom CUDA kernels for arrow drawing, which provides limited
    benefit since OpenCV's implementation is already very fast.
    
    Parameters
    ----------
    Same as draw_vectors_opencv
    
    Returns
    -------
    np.ndarray
        Image with vectors overlaid
    
    Notes
    -----
    If CuPy is not available, automatically falls back to CPU version.
    """
    # Fall back to CPU version
    # GPU arrow drawing would require complex CUDA kernels
    return draw_vectors_opencv(img, x, y, u, v, skip, scale, color, thickness)


def calculate_auto_arrow_scale(u: np.ndarray, v: np.ndarray, 
                               pixel_size: float,
                               target_length_pixels: float = 20.0) -> float:
    """
    Automatically calculate arrow scale for optimal visualization.
    
    This function analyzes the velocity field and calculates a scale
    factor that makes arrows visible but not overwhelming.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity components (can be 2D or 3D with time dimension)
    pixel_size : float
        Physical size per pixel (e.g., microns/pixel)
    target_length_pixels : float, optional
        Target arrow length in pixels for median velocity (default: 20)
    
    Returns
    -------
    float
        Recommended arrow scale factor
    
    Examples
    --------
    >>> # Auto-calibrate arrows for PIV field
    >>> scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)
    >>> print(f"Use arrow_scale={scale:.2f}")
    Use arrow_scale=1.84
    
    >>> # Shorter arrows
    >>> scale = calculate_auto_arrow_scale(u, v, 0.65, target_length_pixels=10)
    
    Notes
    -----
    - Uses median velocity to be robust against outliers
    - Considers pixel size to convert physical velocities to screen pixels
    - Returns 1.0 if median velocity is zero (no motion detected)
    """
    # Calculate velocity magnitude
    velocity_mag = np.sqrt(u**2 + v**2)
    
    # Get median of non-zero velocities
    nonzero_velocities = velocity_mag[velocity_mag > 0]
    if len(nonzero_velocities) == 0:
        return 1.0
    
    median_velocity = np.median(nonzero_velocities)
    
    if median_velocity == 0:
        return 1.0
    
    # Calculate scale to achieve target length
    # velocity is in physical units/time, convert to pixels
    scale = target_length_pixels / (median_velocity / pixel_size)
    
    return scale


def video_with_vectors(stack, 
                      preprocess: Callable, 
                      x: np.ndarray, 
                      y: np.ndarray,
                      u: np.ndarray, 
                      v: np.ndarray,
                      filename: str, 
                      speed: float = 1.0, 
                      skip: int = 1,
                      codec: str = 'h264', 
                      quality: str = 'medium',
                      dt: Optional[float] = None,
                      fps: Optional[float] = None,
                      start: int = 0, 
                      end: int = -1,
                      vector_skip: int = 2, 
                      vector_scale: float = 1.0,
                      vector_color: Tuple[int, int, int] = (255, 255, 0),
                      vector_thickness: int = 2,
                      text: bool = False, 
                      fontsize: float = 2.0,
                      text_color: Tuple[int, int, int] = (255, 255, 255),
                      text_position: Tuple[int, int] = (50, 50),
                      use_gpu: bool = False,
                      verbose: bool = True):
    """
    Create video with vector field overlay (e.g., for PIV visualization).
    
    This function creates a video showing velocity vectors overlaid on
    microscopy images. Much faster than matplotlib-based rendering.
    
    Parameters
    ----------
    stack : object
        Stack object with indexing (.data) and .time() method
    preprocess : callable
        Function to preprocess each frame: preprocess(img) -> normalized_img
    x, y : np.ndarray
        Meshgrid coordinates of vector field (2D arrays)
    u, v : np.ndarray
        Vector components. Shape: (n_frames, n_rows, n_cols) for time-series
        or (n_rows, n_cols) for static field
    filename : str
        Output video filename
    speed : float, optional
        Playback speed multiplier (default: 1.0)
    skip : int, optional
        Frame skip factor (default: 1)
    codec : str, optional
        Video codec (default: 'h264')
    quality : str, optional
        Video quality: 'high', 'medium', 'low' (default: 'medium')
    dt : float, optional
        Time spacing override
    fps : float, optional
        Manual FPS override (ignores speed/dt if set)
    start : int, optional
        Starting frame index (default: 0)
    end : int, optional
        Ending frame index. -1 = all frames (default: -1)
    vector_skip : int, optional
        Draw every N-th vector (default: 2)
    vector_scale : float, optional
        Vector length scaling (default: 1.0)
    vector_color : tuple of int, optional
        Vector color RGB (default: yellow (255, 255, 0))
    vector_thickness : int, optional
        Arrow line thickness (default: 2)
    text : bool, optional
        Add time text overlay (default: False)
    fontsize : float, optional
        Text font size (default: 2.0)
    text_color : tuple of int, optional
        Text color RGB (default: white)
    text_position : tuple of int, optional
        Text position (x, y) pixels (default: (50, 50))
    use_gpu : bool, optional
        Use GPU acceleration if available (default: False)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> # Basic PIV video
    >>> video_with_vectors(stack, preprocess, x, y, u, v, 'piv.mp4')
    
    >>> # High quality with custom arrow style
    >>> video_with_vectors(
    ...     stack, preprocess, x, y, u, v, 'piv_hq.mp4',
    ...     quality='high',
    ...     vector_skip=3,
    ...     vector_scale=2.5,
    ...     vector_color=(0, 255, 0),  # Green arrows
    ...     vector_thickness=3
    ... )
    
    >>> # With time overlay
    >>> video_with_vectors(
    ...     stack, preprocess, x, y, u, v, 'piv_time.mp4',
    ...     text=True,
    ...     fontsize=1.5
    ... )
    
    Notes
    -----
    - For time-series velocity, u and v should have shape (n_frames, rows, cols)
    - For static velocity, u and v should have shape (rows, cols)
    - Increase vector_skip for better performance on large datasets
    - Use calculate_auto_arrow_scale() to determine optimal vector_scale
    """
    # Check if velocity is time-series or static
    is_timeseries = u.ndim == 3
    
    # Get first frame with vectors to determine dimensions
    first_img = preprocess(stack[0].data.astype(float))
    first_img = normalize_image(first_img, output_range='uint8')
    
    # Choose drawing function
    draw_func = draw_vectors_gpu if use_gpu else draw_vectors_opencv
    
    # Get first velocity frame
    u_first = u[0] if is_timeseries else u
    v_first = v[0] if is_timeseries else v
    
    first_frame = draw_func(first_img, x, y, u_first, v_first,
                           skip=vector_skip, scale=vector_scale,
                           color=vector_color, thickness=vector_thickness)
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
        print(f"Vector field: {x.shape}, drawing every {vector_skip} vectors")
    
    # Create video writer
    with VideoWriter(filename, width, height, fps,
                    is_color=True, codec=codec, quality=quality) as writer:
        
        # Process frames
        iterator = range(start, end, skip)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc="Rendering video")
        
        for frame_counter, i in enumerate(iterator):
            # Preprocess frame
            img = preprocess(stack[i].data.astype(float))
            img = normalize_image(img, output_range='uint8')
            
            # Get velocity for this frame
            if is_timeseries:
                idx = min(frame_counter, u.shape[0] - 1)
                u_frame = u[idx]
                v_frame = v[idx]
            else:
                u_frame = u
                v_frame = v
            
            # Draw vectors
            img_with_vectors = draw_func(
                img, x, y, u_frame, v_frame,
                skip=vector_skip, scale=vector_scale, 
                color=vector_color, thickness=vector_thickness
            )
            
            # Add text overlay if requested
            if text:
                label = f't: {times[i]:.2f}s'
                img_with_vectors = add_text_overlay(
                    img_with_vectors, label, text_position,
                    fontsize, text_color
                )
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_with_vectors, cv2.COLOR_RGB2BGR)
            
            # Write frame
            writer.write(img_bgr)
    
    # Report completion
    if verbose:
        file_size_mb = Path(filename).stat().st_size / (1024**2)
        print(f'✓ Video complete: {filename}')
        print(f'  File size: {file_size_mb:.1f} MB')
