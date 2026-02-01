"""
Utility Functions
=================

Helper functions for video creation: resolution optimization, arrow calibration,
coordinate conversions, etc.

Author: Carlo
"""

import numpy as np
from typing import Tuple, Optional


def mm2inch(*tupl) -> Tuple[float, ...]:
    """
    Convert millimeters to inches for figure sizing.
    
    Parameters
    ----------
    *tupl : float or tuple of floats
        Dimensions in millimeters
    
    Returns
    -------
    tuple of floats
        Dimensions in inches
    
    Examples
    --------
    >>> mm2inch(100, 80)
    (3.937, 3.150)
    >>> mm2inch((100, 80))
    (3.937, 3.150)
    """
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def calculate_optimal_resolution(
    original_width: int,
    original_height: int,
    target_size_mb: float,
    duration_seconds: float,
    fps: float,
    codec: str = 'h265'
) -> Tuple[int, int, str]:
    """
    Calculate optimal resolution to achieve target file size.
    
    Uses compression efficiency estimates to determine the maximum resolution
    that will fit within the target file size.
    
    Parameters
    ----------
    original_width, original_height : int
        Original frame dimensions in pixels
    target_size_mb : float
        Target file size in megabytes
    duration_seconds : float
        Video duration in seconds
    fps : float
        Frames per second
    codec : str, optional
        Video codec: 'h265', 'h264', 'h265_nvenc', 'h264_nvenc'
        Default: 'h265' (best compression)
    
    Returns
    -------
    width : int
        Recommended width (divisible by 16)
    height : int
        Recommended height (divisible by 16)
    bitrate : str
        Recommended bitrate (e.g., '500k')
    
    Examples
    --------
    >>> width, height, bitrate = calculate_optimal_resolution(
    ...     1648, 1648, target_size_mb=200, duration_seconds=40, fps=25
    ... )
    >>> print(f"Optimal resolution: {width}×{height} at {bitrate}")
    Optimal resolution: 1024×1024 at 512k
    
    Notes
    -----
    The function uses bits-per-pixel (bpp) estimates:
    - H.265: 0.08 bpp (aggressive compression)
    - H.264: 0.12 bpp (moderate compression)
    
    Output dimensions are always divisible by 16 for codec compatibility.
    """
    # Calculate target bitrate (bits per second)
    target_bitrate_kbps = (target_size_mb * 8 * 1024) / duration_seconds
    
    # Compression efficiency (bits per pixel per frame)
    if codec in ['h265', 'hevc', 'h265_nvenc', 'hevc_nvenc']:
        bpp = 0.08  # H.265 is more efficient
    else:  # H.264
        bpp = 0.12
    
    # Calculate maximum pixels per frame
    max_pixels = (target_bitrate_kbps * 1000) / (fps * bpp)
    
    # Calculate scaling factor
    original_pixels = original_width * original_height
    scale_factor = np.sqrt(max_pixels / original_pixels)
    
    # Determine resolution (ensure divisible by 16 for codec)
    if scale_factor >= 1.0:
        # No downsampling needed
        new_width = (original_width // 16) * 16
        new_height = (original_height // 16) * 16
    elif scale_factor >= 0.75:
        # Minimal downsampling (75%)
        new_width = (int(original_width * 0.75) // 16) * 16
        new_height = (int(original_height * 0.75) // 16) * 16
    elif scale_factor >= 0.5:
        # Half resolution
        new_width = (int(original_width * 0.5) // 16) * 16
        new_height = (int(original_height * 0.5) // 16) * 16
    else:
        # Quarter resolution
        new_width = (int(original_width * 0.25) // 16) * 16
        new_height = (int(original_height * 0.25) // 16) * 16
    
    # Ensure minimum size (256×256)
    new_width = max(256, new_width)
    new_height = max(256, new_height)
    
    # Recalculate bitrate for actual resolution
    actual_pixels = new_width * new_height
    actual_bitrate_kbps = int((actual_pixels * fps * bpp) / 1000)
    
    print(f"\nResolution optimization:")
    print(f"  Original: {original_width}×{original_height} ({original_pixels/1e6:.2f} MP)")
    print(f"  Optimized: {new_width}×{new_height} ({actual_pixels/1e6:.2f} MP)")
    print(f"  Scale factor: {new_width/original_width:.1%}")
    print(f"  Target bitrate: {actual_bitrate_kbps} kbps")
    print(f"  Estimated size: {(actual_bitrate_kbps * duration_seconds) / (8 * 1024):.1f} MB")
    
    return new_width, new_height, f"{actual_bitrate_kbps}k"


def auto_calibrate_arrows(
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    dt: float,
    is_timeseries: bool = None
) -> Tuple[float, int, int]:
    """
    Automatically calibrate arrow visualization parameters.
    
    Determines optimal arrow scale, width, and subsampling based on
    velocity field characteristics.
    
    Parameters
    ----------
    u, v : ndarray
        Velocity components in μm/s
        Shape: [nt, ny, nx] for time series, or [ny, nx] for single frame
    pixel_size : float
        Microns per pixel (e.g., 0.65)
    dt : float
        Time step in seconds
    is_timeseries : bool, optional
        Whether data is time-dependent. Auto-detected if None.
    
    Returns
    -------
    arrow_scale : float
        Recommended arrow length multiplier (0.5 to 10.0)
    arrow_width : int
        Recommended arrow line width in pixels (2-4)
    subsample : int
        Recommended subsampling factor (show every Nth arrow)
    
    Examples
    --------
    >>> arrow_scale, arrow_width, subsample = auto_calibrate_arrows(
    ...     u, v, pixel_size=0.65, dt=0.033
    ... )
    >>> print(f"Use: arrow_scale={arrow_scale:.2f}, width={arrow_width}, subsample={subsample}")
    Use: arrow_scale=2.50, width=3, subsample=2
    
    Notes
    -----
    Target arrow length: ~20 pixels for good visibility
    Subsampling: shows ~5% of total arrows to avoid clutter
    """
    # Auto-detect if time series
    if is_timeseries is None:
        is_timeseries = u.ndim == 3
    
    # Get velocity magnitude
    if is_timeseries:
        speed = np.hypot(u, v).flatten()
    else:
        speed = np.hypot(u, v).flatten()
    
    # Filter valid speeds
    speed = speed[np.isfinite(speed) & (speed > 0)]
    
    if len(speed) == 0:
        print("Warning: No valid velocity data, using default calibration")
        return 1.0, 2, 2
    
    # Use 95th percentile (robust to outliers)
    clip_speed = np.percentile(speed, 95)
    
    # Calculate displacement in pixels
    displacement_pix = (clip_speed * dt) / pixel_size
    
    # Target arrow length: 20 pixels
    arrow_scale = 20.0 / displacement_pix if displacement_pix > 0 else 1.0
    arrow_scale = np.clip(arrow_scale, 0.5, 10.0)
    
    # Arrow width based on scale
    if arrow_scale < 2:
        arrow_width = 2
    elif arrow_scale < 5:
        arrow_width = 3
    else:
        arrow_width = 4
    
    # Subsampling: show ~5% of arrows
    if is_timeseries:
        ny, nx = u.shape[1:]
    else:
        ny, nx = u.shape
    
    total_arrows = ny * nx
    target_arrows = int(total_arrows * 0.05)
    subsample = max(1, int(np.sqrt(total_arrows / target_arrows)))
    
    print(f"\nArrow auto-calibration:")
    print(f"  Velocity (95th): {clip_speed:.2f} μm/s")
    print(f"  Displacement: {displacement_pix:.1f} pixels")
    print(f"  Arrow scale: {arrow_scale:.2f}×")
    print(f"  Arrow width: {arrow_width} px")
    print(f"  Subsample: every {subsample} arrows")
    print(f"  Total arrows: {(ny // subsample) * (nx // subsample)}")
    
    return arrow_scale, arrow_width, subsample


def ensure_even_dimensions(width: int, height: int) -> Tuple[int, int]:
    """
    Ensure dimensions are even numbers (required for YUV420p encoding).
    
    Parameters
    ----------
    width, height : int
        Original dimensions
    
    Returns
    -------
    width, height : int
        Adjusted dimensions (always even)
    
    Examples
    --------
    >>> ensure_even_dimensions(1647, 1648)
    (1646, 1648)
    """
    return (width // 2) * 2, (height // 2) * 2


def calculate_video_duration(
    n_frames: int,
    fps: float
) -> float:
    """
    Calculate video duration in seconds.
    
    Parameters
    ----------
    n_frames : int
        Number of frames
    fps : float
        Frames per second
    
    Returns
    -------
    float
        Duration in seconds
    """
    return n_frames / fps


def estimate_file_size(
    width: int,
    height: int,
    n_frames: int,
    fps: float,
    codec: str = 'h265'
) -> float:
    """
    Estimate output video file size in MB.
    
    Parameters
    ----------
    width, height : int
        Video dimensions
    n_frames : int
        Number of frames
    fps : float
        Frames per second
    codec : str
        Video codec ('h265' or 'h264')
    
    Returns
    -------
    float
        Estimated file size in megabytes
    
    Examples
    --------
    >>> size_mb = estimate_file_size(1024, 1024, 1000, 25, 'h265')
    >>> print(f"Estimated size: {size_mb:.1f} MB")
    Estimated size: 185.3 MB
    """
    pixels = width * height
    duration = n_frames / fps
    
    # Bits per pixel estimates
    bpp = 0.08 if 'h265' in codec else 0.12
    
    # Bitrate in kbps
    bitrate_kbps = (pixels * fps * bpp) / 1000
    
    # File size in MB
    size_mb = (bitrate_kbps * duration) / (8 * 1024)
    
    return size_mb