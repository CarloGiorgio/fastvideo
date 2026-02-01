"""
Vector Field Video Creation
===========================

Functions for rendering velocity fields and vector overlays on videos.
"""

import numpy as np
import cv2
import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple

from .video_writer import VideoWriter
from .video_utils import calculate_fps, normalize_image, add_text_overlay, validate_frame_range

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
    Draw vector field on image using OpenCV (much faster than matplotlib).
    
    Parameters
    ----------
    img : np.ndarray
        Background image (grayscale or RGB)
    x, y : np.ndarray
        Meshgrid coordinates of vector field
    u, v : np.ndarray
        Vector components (same shape as x, y)
    skip : int, optional
        Draw every N-th vector (default: 2)
    scale : float, optional
        Vector length scaling (default: 1.0)
    color : tuple, optional
        Arrow color RGB (default: yellow)
    thickness : int, optional
        Arrow thickness (default: 2)
    arrow_tip : float, optional
        Arrow tip size ratio (default: 0.3)
    
    Returns
    -------
    np.ndarray
        Image with vectors overlaid (RGB uint8)
    
    Examples
    --------
    >>> img_gray = np.random.rand(512, 512)
    >>> x, y = np.meshgrid(np.arange(512), np.arange(512))
    >>> u = np.ones_like(x) * 5
    >>> v = np.ones_like(y) * 3
    >>> img_with_arrows = draw_vectors_opencv(img_gray, x, y, u, v, skip=16)
    """
    # Convert to RGB if grayscale
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()
    
    # Ensure uint8
    if img_rgb.dtype != np.uint8:
        img_rgb = normalize_image(img_rgb)
    
    # Draw arrows (subsample for speed)
    for i in range(0, x.shape[0], skip):
        for j in range(0, x.shape[1], skip):
            x0, y0 = int(x[i, j]), int(y[i, j])
            dx, dy = u[i, j] * scale, v[i, j] * scale
            x1, y1 = int(x0 + dx), int(y0 + dy)
            
            # Skip very small vectors
            if abs(dx) < 0.5 and abs(dy) < 0.5:
                continue
            
            # Skip if out of bounds
            if not (0 <= x0 < img_rgb.shape[1] and 0 <= y0 < img_rgb.shape[0]):
                continue
            
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
    
    Currently falls back to CPU version. GPU acceleration for arrow
    drawing is complex and provides limited benefit for typical use cases.
    
    Parameters: Same as draw_vectors_opencv
    
    Returns
    -------
    np.ndarray
        Image with vectors overlaid
    """
    if not HAS_CUPY:
        return draw_vectors_opencv(img, x, y, u, v, skip, scale, color, thickness)
    
    # For now, use CPU version
    # GPU arrow drawing would require custom CUDA kernels
    return draw_vectors_opencv(img, x, y, u, v, skip, scale, color, thickness)


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
                      text_position: Tuple[int, int] = (100, 100),
                      use_gpu: bool = False):
    """
    Create video with vector field overlay (e.g., for PIV visualization).
    
    Much faster than matplotlib-based rendering.
    
    Parameters
    ----------
    stack : object
        Stack object with indexing and .time() method
    preprocess : callable
        Function to preprocess each frame: preprocess(img) -> normalized_img
    x, y : np.ndarray
        Meshgrid coordinates of vector field (2D arrays)
    u, v : np.ndarray
        Vector components with shape (n_frames, n_rows, n_cols)
    filename : str
        Output video filename
    speed : float, optional
        Playback speed multiplier (default: 1.0)
    skip : int, optional
        Frame skip factor (default: 1)
    codec : str, optional
        Video codec (default: 'h264')
    quality : str, optional
        Video quality (default: 'medium')
    dt : float, optional
        Time spacing override
    fps : float, optional
        Manual FPS override
    start : int, optional
        Starting frame index (default: 0)
    end : int, optional
        Ending frame index (default: -1)
    vector_skip : int, optional
        Draw every N-th vector (default: 2)
    vector_scale : float, optional
        Vector length scaling (default: 1.0)
    vector_color : tuple, optional
        Vector color RGB (default: yellow)
    vector_thickness : int, optional
        Arrow line thickness (default: 2)
    text : bool, optional
        Add time text overlay (default: False)
    fontsize : float, optional
        Text font size (default: 2.0)
    text_color : tuple, optional
        Text color RGB (default: white)
    text_position : tuple, optional
        Text position (x, y) (default: (100, 100))
    use_gpu : bool, optional
        Use GPU acceleration if available (default: False)
    
    Examples
    --------
    >>> # Create PIV video
    >>> video_with_vectors(stack, preprocess, x, y, u_field, v_field,
    ...                   'piv_video.mp4', vector_skip=3, vector_scale=2.0)
    """
    # Get first frame with vectors
    first_img = preprocess(stack[0].data.astype(float))
    first_img = normalize_image(first_img)
    
    # Choose drawing function
    draw_func = draw_vectors_gpu if use_gpu else draw_vectors_opencv
    
    first_frame = draw_func(first_img, x, y, u[0], v[0],
                           skip=vector_skip, scale=vector_scale,
                           color=vector_color, thickness=vector_thickness)
    height, width = first_frame.shape[:2]
    
    # Calculate FPS
    times = stack.time()
    if fps is None:
        fps = calculate_fps(times, speed, skip, dt)
    
    # Handle frame range
    start, end = validate_frame_range(start, end, len(stack))
    
    print(f"Creating vector video: {end - start} frames (skip={skip})")
    print(f"Vector field: {x.shape}, drawing every {vector_skip} vectors")
    print(f"FPS: {fps:.1f}, codec: {codec}")
    
    # Create video writer
    with VideoWriter(filename, width, height, fps,
                    is_color=True, codec=codec, quality=quality) as writer:
        
        for i in tqdm.tqdm(range(start, end, skip), desc="Rendering"):
            # Preprocess frame
            img = preprocess(stack[i].data.astype(float))
            img = normalize_image(img)
            
            # Draw vectors
            img_with_vectors = draw_func(
                img, x, y, u[i], v[i],
                skip=vector_skip, scale=vector_scale, 
                color=vector_color, thickness=vector_thickness
            )
            
            # Add text overlay if requested
            if text:
                label = f't:{times[i]:.2f}s'
                img_with_vectors = add_text_overlay(
                    img_with_vectors, label, text_position,
                    fontsize, text_color
                )
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_with_vectors, cv2.COLOR_RGB2BGR)
            
            # Write frame
            writer.write(img_bgr)
    
    print('Video complete!')
    file_size_mb = Path(filename).stat().st_size / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")


def calculate_auto_arrow_scale(u: np.ndarray, v: np.ndarray, 
                               pixel_size: float,
                               target_length_pixels: float = 20.0) -> float:
    """
    Automatically calculate arrow scale for good visualization.
    
    Parameters
    ----------
    u, v : np.ndarray
        Velocity components
    pixel_size : float
        Physical size per pixel (e.g., microns/pixel)
    target_length_pixels : float, optional
        Target arrow length in pixels (default: 20)
    
    Returns
    -------
    float
        Recommended arrow scale factor
    
    Examples
    --------
    >>> scale = calculate_auto_arrow_scale(u, v, 0.65, target_length_pixels=15)
    >>> print(f"Use arrow_scale={scale:.2f}")
    """
    # Calculate median velocity magnitude
    velocity_mag = np.sqrt(u**2 + v**2)
    median_velocity = np.median(velocity_mag[velocity_mag > 0])
    
    if median_velocity == 0:
        return 1.0
    
    # Calculate scale to achieve target length
    # velocity is in physical units/time, we want pixels
    scale = target_length_pixels / (median_velocity / pixel_size)
    
    return scale
