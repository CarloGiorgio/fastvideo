"""
Video Creation Functions
========================

Main functions for creating velocity field videos with validation
and error handling.

Author: Carlo
"""

import numpy as np
import cv2
import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple

from .ffmpeg_writer import RobustFFmpegWriter, check_ffmpeg_codecs
from .overlay import VelocityOverlayProcessor, GPUVelocityOverlayProcessor
from .utils import calculate_optimal_resolution, ensure_even_dimensions
from .preview import validate_video_settings


def create_video(
    stack,
    preprocessor: Callable,
    output_path: str,
    target_size_mb: float = 200,
    dt: Optional[float] = None,
    fps: float = 25.0,
    codec: str = 'h264',
    preset: str = 'medium',
    resolution_scale: Optional[float] = None,
    start: int = 0,
    end: Optional[int] = None,
    skip: int = 1,
    text: bool = False,
    text_position: Tuple[int, int] = (50, 50),
    text_size: float = 1.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    auto_optimize_resolution: bool = True,
    use_gpu: bool = False,
    verbose: bool = False
    ):
    times = stack.time()
    if dt is None:
        dt = np.mean(np.diff(times[1:]))
    
    # Handle frame range
    if end is None or end <= 0:
        end = len(stack)
    end = min(end, len(stack))
    
    # Calculate duration
    n_frames = (end - start) // skip
    duration = n_frames / fps
    
    # Get original dimensions
    first_img = preprocessor(stack[0].data.astype(float))
    
    # Handle CuPy arrays
    try:
        if hasattr(first_img, 'get'):
            first_img = first_img.get()
    except:
        pass
    
    original_height, original_width = first_img.shape
    
    # Determine output resolution
    if auto_optimize_resolution and resolution_scale is None:
        opt_width, opt_height, bitrate = calculate_optimal_resolution(
            original_width, original_height,
            target_size_mb, duration, fps, codec
        )
        output_resolution = (opt_width, opt_height)
    elif resolution_scale is not None:
        # Manual scaling
        opt_width = int(original_width * resolution_scale)
        opt_height = int(original_height * resolution_scale)
        opt_width, opt_height = ensure_even_dimensions(opt_width, opt_height)
        output_resolution = (opt_width, opt_height)
        
        # Calculate bitrate
        from .utils import estimate_file_size
        estimated_size = estimate_file_size(opt_width, opt_height, n_frames, fps, codec)
        bitrate_kbps = int((estimated_size * 8 * 1024) / duration)
        bitrate = f"{bitrate_kbps}k"
    else:
        # No downsampling
        opt_width, opt_height = ensure_even_dimensions(original_width, original_height)
        output_resolution = None if (opt_width, opt_height) == (original_width, original_height) else (opt_width, opt_height)
        bitrate = '500k'


    if text:
        time_array = times[start:end:skip]
    
    # Print summary
    print("\n" + "="*70)
    print("VIDEO CREATION")
    print("="*70)
    print(f"  Processing: {'GPU' if use_gpu else 'CPU'}")
    print(f"  Frames: {start} to {end}, skip={skip} ({n_frames} total)")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Original: {original_width}×{original_height}")
    print(f"  Output: {opt_width}×{opt_height}")
    print(f"  Target size: {target_size_mb} MB")
    print(f"  Codec: {codec}, Preset: {preset}")
    print(f"  Bitrate: {bitrate}")
    print("="*70 + "\n")
    
    # Create FFmpeg writer
    with RobustFFmpegWriter(
        output_path,
        opt_width,
        opt_height,
        fps,
        bitrate=bitrate,
        codec=codec,
        preset=preset,
        verbose=verbose
    ) as writer:
        
        # Process frames
        for i, frame_idx in enumerate(tqdm.tqdm(
            range(start, end, skip),
            desc="Rendering video"
        )):
            # Get frame with overlay
            img = preprocessor(stack[frame_idx].data.astype(float))

            # Handle CuPy arrays
            try:
                if hasattr(img, 'get'):
                    img = img.get()
            except:
                pass

            # Normalize float [0, 1] to uint8 [0, 255]
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Ensure correct dimensions
            if img.shape[:2] != (opt_height, opt_width):
                img = cv2.resize(
                    img,
                    (opt_width, opt_height),
                    interpolation=cv2.INTER_AREA
                )
            
            # Add text overlay
            if text:
                label = f't: {time_array[i]:.2f}s'
                img = cv2.putText(
                    img, label, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    text_color, 2, cv2.LINE_AA
                )
            
            # Write frame
            writer.write(img)
    
    print("\n✓ Video creation complete!")

def create_velocity_video(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_path: str,
    target_size_mb: float = 200,
    dt: Optional[float] = None,
    fps: float = 25.0,
    codec: str = 'h264',
    preset: str = 'medium',
    resolution_scale: Optional[float] = None,
    arrow_scale: Optional[float] = None,
    arrow_width: int = 2,
    arrow_color: Tuple[int, int, int] = (255, 255, 0),
    subsample: Optional[int] = None,
    start: int = 0,
    end: Optional[int] = None,
    skip: int = 1,
    text: bool = False,
    text_position: Tuple[int, int] = (50, 50),
    text_size: float = 1.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    auto_optimize_resolution: bool = True,
    use_gpu: bool = False,
    verbose: bool = False
):
    """
    Create ultra-compressed video with velocity field overlay.
    
    Main function for video creation with automatic optimization,
    robust error handling, and progress tracking.
    
    Parameters
    ----------
    stack : object
        Image stack with .data and .time() methods
    preprocessor : callable
        Image preprocessing function: preprocess(img) -> normalized_img
        Output should be in [0, 1] range or uint8 [0, 255]
    x, y : ndarray
        Velocity grid coordinates in microns, shape [ny, nx]
    u, v : ndarray
        Velocity components in μm/s
        Shape: [nt, ny, nx] for time series, or [ny, nx] for single frame
    pixel_size : float
        Microns per pixel (e.g., 0.65 for typical microscopy)
    output_path : str
        Output video file path (e.g., 'bacteria.mp4')
    target_size_mb : float, optional
        Target file size in megabytes (default: 200)
        Automatically adjusts resolution to meet target
    fps : float, optional
        Frames per second (default: 25)
    codec : str, optional
        Video codec: 'h264', 'h265', 'h264_nvenc', 'h265_nvenc'
        Default: 'h264' (most compatible)
        h265 provides better compression (~30% smaller files)
        *_nvenc requires NVIDIA GPU with hardware encoding
    preset : str, optional
        Encoding preset:
        - CPU: 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
        - NVENC: 'p1' (fast) to 'p7' (slow, best quality)
        Default: 'medium'
    resolution_scale : float, optional
        Manual resolution scaling (0.5 = half, 0.25 = quarter)
        If None and auto_optimize_resolution=True, calculated automatically
    arrow_scale : float, optional
        Arrow length multiplier (None = auto-calibrate to ~20 pixels)
    arrow_width : int, optional
        Arrow line width in pixels (default: 2)
    arrow_color : tuple, optional
        RGB arrow color, 0-255 (default: yellow = (255, 255, 0))
    subsample : int, optional
        Show every Nth arrow (None = auto-calibrate to ~5% density)
    start : int, optional
        Starting frame index (default: 0)
    end : int, optional
        Ending frame index (None = all frames)
    skip : int, optional
        Frame skip factor (default: 1 = all frames)
        Use skip=2 for half the frames (smaller file, lower temporal resolution)
    text : bool, optional
        Add time overlay (default: False)
    text_position : tuple, optional
        Text position (x, y) in pixels (default: (50, 50))
    text_size : float, optional
        Text size multiplier (default: 1.5)
    text_color : tuple, optional
        Text RGB color (default: white = (255, 255, 255))
    auto_optimize_resolution : bool, optional
        Automatically calculate optimal resolution for target size (default: True)
    use_gpu : bool, optional
        Use GPU acceleration (requires CuPy and NVIDIA GPU) (default: False)
    verbose : bool, optional
        Print detailed FFmpeg output for debugging (default: False)
    
    Returns
    -------
    None
        Video file is written to disk
    
    Examples
    --------
    >>> from slmicro_video import create_velocity_video
    >>> from scipy.ndimage import gaussian_filter
    >>> import numpy as np
    >>> 
    >>> # Load data
    >>> stack = loadstack('bacteria.tif')
    >>> vel = np.load('velocity.npz')
    >>> 
    >>> # Define preprocessing
    >>> def preprocess(img):
    ...     img = gaussian_filter(img, 1.0)
    ...     return (img - img.min()) / img.ptp()
    >>> 
    >>> # Create video (automatic optimization)
    >>> create_velocity_video(
    ...     stack, preprocess,
    ...     vel['x'], vel['y'], vel['u'], vel['v'],
    ...     pixel_size=0.65,
    ...     output_path='bacteria.mp4',
    ...     target_size_mb=200,
    ...     text=True
    ... )
    
    >>> # High quality (larger file, slower encoding)
    >>> create_velocity_video(
    ...     stack, preprocess,
    ...     vel['x'], vel['y'], vel['u'], vel['v'],
    ...     pixel_size=0.65,
    ...     output_path='bacteria_hq.mp4',
    ...     codec='h265',
    ...     preset='slow',
    ...     target_size_mb=300
    ... )
    
    >>> # GPU-accelerated (fastest)
    >>> create_velocity_video(
    ...     stack, preprocess,
    ...     vel['x'], vel['y'], vel['u'], vel['v'],
    ...     pixel_size=0.65,
    ...     output_path='bacteria_gpu.mp4',
    ...     codec='h265_nvenc',
    ...     use_gpu=True
    ... )
    
    Notes
    -----
    The function automatically:
    - Adjusts resolution to meet target file size
    - Calibrates arrow scaling for visibility
    - Handles dimension requirements (even numbers)
    - Provides progress tracking
    - Reports final file size
    
    For debugging, use verbose=True to see FFmpeg output.
    """
    # Calculate dt
    times = stack.time()
    if dt is None:
        dt = np.mean(np.diff(times[1:]))
    
    # Handle frame range
    if end is None or end <= 0:
        end = len(stack)
    end = min(end, len(stack))
    
    # Calculate duration
    n_frames = (end - start) // skip
    duration = n_frames / fps
    
    # Get original dimensions
    first_img = preprocessor(stack[0].data.astype(float))
    
    # Handle CuPy arrays
    try:
        if hasattr(first_img, 'get'):
            first_img = first_img.get()
    except:
        pass
    
    original_height, original_width = first_img.shape
    
    # Determine output resolution
    if auto_optimize_resolution and resolution_scale is None:
        opt_width, opt_height, bitrate = calculate_optimal_resolution(
            original_width, original_height,
            target_size_mb, duration, fps, codec
        )
        output_resolution = (opt_width, opt_height)
    elif resolution_scale is not None:
        # Manual scaling
        opt_width = int(original_width * resolution_scale)
        opt_height = int(original_height * resolution_scale)
        opt_width, opt_height = ensure_even_dimensions(opt_width, opt_height)
        output_resolution = (opt_width, opt_height)
        
        # Calculate bitrate
        from .utils import estimate_file_size
        estimated_size = estimate_file_size(opt_width, opt_height, n_frames, fps, codec)
        bitrate_kbps = int((estimated_size * 8 * 1024) / duration)
        bitrate = f"{bitrate_kbps}k"
    else:
        # No downsampling
        opt_width, opt_height = ensure_even_dimensions(original_width, original_height)
        output_resolution = None if (opt_width, opt_height) == (original_width, original_height) else (opt_width, opt_height)
        bitrate = '500k'
    
    # Create processor
    if use_gpu:
        try:
            processor = GPUVelocityOverlayProcessor(
                preprocessor, x, y, u, v, pixel_size, dt,
                output_resolution=None,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                arrow_color=arrow_color,
                subsample=subsample
            )
            print("Using GPU acceleration")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU processing")
            use_gpu = False
            processor = VelocityOverlayProcessor(
                preprocessor, x, y, u, v, pixel_size, dt,
                output_resolution=None,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                arrow_color=arrow_color,
                subsample=subsample
            )
    else:
        processor = VelocityOverlayProcessor(
            preprocessor, x, y, u, v, pixel_size, dt,
            output_resolution=None,
            arrow_scale=arrow_scale,
            arrow_width=arrow_width,
            arrow_color=arrow_color,
            subsample=subsample
        )
    
    # Get time array
    if text:
        time_array = times[start:end:skip]
    
    # Print summary
    print("\n" + "="*70)
    print("VIDEO CREATION")
    print("="*70)
    print(f"  Processing: {'GPU' if use_gpu else 'CPU'}")
    print(f"  Frames: {start} to {end}, skip={skip} ({n_frames} total)")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Original: {original_width}×{original_height}")
    print(f"  Output: {opt_width}×{opt_height}")
    print(f"  Target size: {target_size_mb} MB")
    print(f"  Codec: {codec}, Preset: {preset}")
    print(f"  Bitrate: {bitrate}")
    print("="*70 + "\n")
    
    # Create FFmpeg writer
    with RobustFFmpegWriter(
        output_path,
        opt_width,
        opt_height,
        fps,
        bitrate=bitrate,
        codec=codec,
        preset=preset,
        verbose=verbose
    ) as writer:
        
        # Process frames
        for i, frame_idx in enumerate(tqdm.tqdm(
            range(start, end, skip),
            desc="Rendering video"
        )):
            # Get frame with overlay
            img_with_arrows = processor(
                stack[frame_idx].data.astype(float),
                frame_idx
            )
            
            # Ensure correct dimensions
            if img_with_arrows.shape[:2] != (opt_height, opt_width):
                img_with_arrows = cv2.resize(
                    img_with_arrows,
                    (opt_width, opt_height),
                    interpolation=cv2.INTER_AREA
                )
            
            # Add text overlay
            if text:
                label = f't: {time_array[i]:.2f}s'
                img_with_arrows = cv2.putText(
                    img_with_arrows, label, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    text_color, 2, cv2.LINE_AA
                )
            
            # Write frame
            writer.write(img_with_arrows)
    
    print("\n✓ Video creation complete!")


def create_video_simple(
    stack,
    preprocessor: Callable,
    velocity_file: str,
    output_path: str,
    pixel_size: float = 0.65,
    target_size_mb: float = 200,
    dt: float =1/25,
    fps: float = 25,
    skip: int = 1,
    text: bool = True,
    use_gpu: bool = False
):
    """
    Simplified interface for common use case.
    
    Loads velocity data from .npz file and creates video with sensible defaults.
    
    Parameters
    ----------
    stack : object
        Image stack
    preprocessor : callable
        Preprocessing function
    velocity_file : str
        Path to .npz file with 'x', 'y', 'u', 'v' arrays
    output_path : str
        Output video path
    pixel_size : float, optional
        Microns per pixel (default: 0.65)
    target_size_mb : float, optional
        Target file size in MB (default: 200)
    fps : float, optional
        Frames per second (default: 25)
    skip : int, optional
        Frame skip factor (default: 1)
    text : bool, optional
        Add time overlay (default: True)
    use_gpu : bool, optional
        Use GPU acceleration (default: False)
    
    Examples
    --------
    >>> from slmicro_video import create_video_simple
    >>> from cam import loadstack
    >>> from scipy.ndimage import gaussian_filter
    >>> 
    >>> stack = loadstack('bacteria.tif')
    >>> preprocess = lambda img: gaussian_filter(img, 1) / img.max()
    >>> 
    >>> create_video_simple(
    ...     stack, preprocess,
    ...     velocity_file='velocity.npz',
    ...     output_path='bacteria.mp4',
    ...     target_size_mb=200
    ... )
    """
    # Load velocity data
    vel = np.load(velocity_file)
    
    # Try GPU encoding first, fallback to CPU
    try:
        create_velocity_video(
            stack, preprocessor,
            vel['x'], vel['y'], vel['u'], vel['v'],
            dt = dt,
            pixel_size=pixel_size,
            output_path=output_path,
            target_size_mb=target_size_mb,
            fps=fps,
            codec='h265_nvenc' if use_gpu else 'h265',
            skip=skip,
            text=text,
            use_gpu=use_gpu
        )
    except Exception as e:
        if 'nvenc' in str(e).lower() or 'gpu' in str(e).lower():
            print(f"\nGPU encoding failed: {e}")
            print("Retrying with CPU encoding...")
            
            create_velocity_video(
                stack, preprocessor,
                vel['x'], vel['y'], vel['u'], vel['v'],
                dt = dt,
                pixel_size=pixel_size,
                output_path=output_path,
                target_size_mb=target_size_mb,
                fps=fps,
                codec='h264',  # Most compatible
                skip=skip,
                text=text,
                use_gpu=False
            )
        else:
            raise


def create_video_with_validation(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_path: str,
    target_size_mb: float = 200,
    dt : float =1/25,
    fps: float = 25.0,
    codec: str = 'h264',
    **kwargs
) -> Optional[dict]:
    """
    Complete workflow: validate → preview → create video.
    
    Validates all settings before creating video to catch issues early.
    Shows preview images and asks for confirmation if warnings are present.
    
    Parameters
    ----------
    Same as create_velocity_video
    
    Returns
    -------
    dict or None
        Validation report if successful, None if cancelled or failed
    
    Examples
    --------
    >>> # Automatic validation before creating video
    >>> report = create_video_with_validation(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     output_path='bacteria.mp4',
    ...     target_size_mb=200
    ... )
    >>> 
    >>> if report:
    ...     print("✓ Video created successfully")
    ... else:
    ...     print("✗ Video creation cancelled or failed")
    """
    print("="*70)
    print("VIDEO CREATION WITH VALIDATION")
    print("="*70)
    
    # Step 1: Validate settings
    print("\nStep 1: Validating settings...")
    report = validate_video_settings(
        stack, preprocessor, x, y, u, v, pixel_size,
        target_size_mb=target_size_mb,
        fps=fps,
        codec=codec,
        **kwargs
    )
    
    if not report['is_valid']:
        print("\n✗ Validation failed. Please fix errors and try again.")
        return None
    
    # Step 2: Ask user to confirm
    if report['warnings']:
        print("\n⚠ There are warnings. Do you want to proceed? (y/n)")
        try:
            response = input().strip().lower()
            if response != 'y':
                print("Cancelled by user.")
                return None
        except:
            print("\nProceeding without confirmation (running in non-interactive mode)")
    
    # Step 3: Create video
    print("\nStep 2: Creating video...")
    
    create_velocity_video(
        stack, preprocessor, x, y, u, v, pixel_size,
        dt = dt,
        output_path=output_path,
        target_size_mb=target_size_mb,
        fps=fps,
        codec=codec,
        **kwargs
    )
    
    print("\n✓ Video creation complete!")
    return report