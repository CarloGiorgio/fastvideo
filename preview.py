"""
Preview and Validation Functions
=================================

Functions to preview and validate video settings before rendering.
Helps identify issues with preprocessing, arrow visibility, and file size.

Author: Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict
from matplotlib.widgets import Slider


def preview_single_frame(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    frame_idx: int = 0,
    output_resolution: Optional[Tuple[int, int]] = None,
    arrow_scale: Optional[float] = None,
    arrow_width: int = 2,
    arrow_color: Tuple[int, int, int] = (255, 255, 0),
    subsample: Optional[int] = None,
    save_path: Optional[str] = None,
    use_gpu: bool = False
) -> np.ndarray:
    """
    Preview a single frame with velocity overlay.
    
    Shows exactly what the video will look like before rendering.
    
    Parameters
    ----------
    stack : object
        Image stack with .data and .time() attributes
    preprocessor : callable
        Image preprocessing function
    x, y : ndarray
        Velocity grid coordinates in microns [ny, nx]
    u, v : ndarray
        Velocity components in μm/s [nt, ny, nx] or [ny, nx]
    pixel_size : float
        Microns per pixel
    frame_idx : int, optional
        Which frame to preview (default: 0)
    output_resolution : tuple, optional
        Target resolution (width, height)
    arrow_scale : float, optional
        Arrow scaling factor (None = auto)
    arrow_width : int, optional
        Arrow line width (default: 2)
    arrow_color : tuple, optional
        RGB arrow color (default: yellow)
    subsample : int, optional
        Arrow subsampling (None = auto)
    save_path : str, optional
        Save preview to file
    use_gpu : bool, optional
        Use GPU processing (default: False)
    
    Returns
    -------
    ndarray
        Preview image (H, W, 3) uint8
    
    Examples
    --------
    >>> # Quick preview
    >>> preview = preview_single_frame(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     frame_idx=100
    ... )
    >>> plt.figure(figsize=(10, 10))
    >>> plt.imshow(preview)
    >>> plt.title('Video Preview')
    >>> plt.axis('off')
    >>> plt.show()
    
    >>> # Save preview
    >>> preview_single_frame(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     frame_idx=100,
    ...     save_path='preview.png'
    ... )
    """
    # Calculate dt
    times = stack.time()
    dt = np.mean(np.diff(times[1:]))
    
    # Select processor
    if use_gpu:
        from .overlay import GPUVelocityOverlayProcessor as ProcessorClass
    else:
        from .overlay import VelocityOverlayProcessor as ProcessorClass
    
    # Create processor
    processor = ProcessorClass(
        preprocessor, x, y, u, v, pixel_size, dt,
        output_resolution=output_resolution,
        arrow_scale=arrow_scale,
        arrow_width=arrow_width,
        arrow_color=arrow_color,
        subsample=subsample
    )
    
    # Get frame
    img = stack[frame_idx].data.astype(float)
    preview = processor(img, frame_idx)
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        print(f"✓ Preview saved to: {save_path}")
    
    return preview


def preview_multiple_frames(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    frame_indices: Optional[List[int]] = None,
    n_frames: int = 4,
    output_resolution: Optional[Tuple[int, int]] = None,
    arrow_scale: Optional[float] = None,
    arrow_width: int = 2,
    arrow_color: Tuple[int, int, int] = (255, 255, 0),
    subsample: Optional[int] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
    use_gpu: bool = False
):
    """
    Preview multiple frames to check consistency.
    
    Displays a grid of frames to verify that velocity overlay
    looks good throughout the video.
    
    Parameters
    ----------
    stack : object
        Image stack
    preprocessor : callable
        Preprocessing function
    x, y, u, v : ndarray
        Velocity data
    pixel_size : float
        Microns per pixel
    frame_indices : list, optional
        Specific frames to preview (default: evenly spaced)
    n_frames : int, optional
        Number of frames to show if frame_indices not specified (default: 4)
    output_resolution : tuple, optional
        Target resolution
    arrow_scale, arrow_width, arrow_color, subsample : optional
        Arrow parameters
    figsize : tuple, optional
        Figure size (default: (16, 8))
    save_path : str, optional
        Save preview grid to file
    use_gpu : bool, optional
        Use GPU processing
    
    Examples
    --------
    >>> # Preview 6 evenly-spaced frames
    >>> preview_multiple_frames(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     n_frames=6
    ... )
    
    >>> # Preview specific frames
    >>> preview_multiple_frames(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     frame_indices=[0, 100, 500, 999],
    ...     save_path='preview_grid.png'
    ... )
    """
    # Determine which frames to preview
    if frame_indices is None:
        total_frames = len(stack)
        frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int).tolist()
    
    # Calculate dt
    times = stack.time()
    dt = np.mean(np.diff(times[1:]))
    
    # Create processor
    if use_gpu:
        from .overlay import GPUVelocityOverlayProcessor as ProcessorClass
    else:
        from .overlay import VelocityOverlayProcessor as ProcessorClass
    
    processor = ProcessorClass(
        preprocessor, x, y, u, v, pixel_size, dt,
        output_resolution=output_resolution,
        arrow_scale=arrow_scale,
        arrow_width=arrow_width,
        arrow_color=arrow_color,
        subsample=subsample
    )
    
    # Create figure
    n_cols = min(4, len(frame_indices))
    n_rows = (len(frame_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Process each frame
    print(f"Generating preview for {len(frame_indices)} frames...")
    for idx, frame_idx in enumerate(frame_indices):
        img = stack[frame_idx].data.astype(float)
        preview = processor(img, frame_idx)
        
        axes[idx].imshow(preview)
        axes[idx].set_title(f'Frame {frame_idx}\nt = {times[frame_idx]:.2f}s')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(frame_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Preview grid saved to: {save_path}")
    
    plt.show()


def validate_video_settings(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_resolution: Optional[Tuple[int, int]] = None,
    arrow_scale: Optional[float] = None,
    arrow_width: int = 2,
    arrow_color: Tuple[int, int, int] = (255, 255, 0),
    subsample: Optional[int] = None,
    target_size_mb: float = 200,
    fps: float = 25,
    codec: str = 'h265',
    start: int = 0,
    end: Optional[int] = None,
    skip: int = 1,
    use_gpu: bool = False
) -> Dict:
    """
    Validate all video settings and provide detailed diagnostics.
    
    Performs comprehensive checks on:
    - Image preprocessing (brightness, contrast)
    - Arrow visibility and scaling
    - Resolution and file size estimates
    - Velocity field coverage
    - Frame quality consistency
    
    Parameters
    ----------
    stack : object
        Image stack
    preprocessor : callable
        Preprocessing function
    x, y, u, v : ndarray
        Velocity data
    pixel_size : float
        Microns per pixel
    output_resolution : tuple, optional
        Target resolution
    arrow_scale, arrow_width, arrow_color, subsample : optional
        Arrow parameters
    target_size_mb : float, optional
        Target file size in MB (default: 200)
    fps : float, optional
        Frames per second (default: 25)
    codec : str, optional
        Video codec (default: 'h265')
    start, end, skip : int, optional
        Frame range parameters
    use_gpu : bool, optional
        Use GPU processing
    
    Returns
    -------
    dict
        Validation report with the following keys:
        - 'is_valid': bool - Overall validation status
        - 'warnings': list - Warning messages
        - 'errors': list - Error messages (must fix)
        - 'info': dict - Diagnostic information
        - 'recommendations': list - Suggested improvements
    
    Examples
    --------
    >>> # Validate settings
    >>> report = validate_video_settings(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     target_size_mb=200,
    ...     fps=25
    ... )
    >>> 
    >>> # Check if valid
    >>> if report['is_valid']:
    ...     print("✓ Settings are valid!")
    ...     # Proceed with video creation
    ... else:
    ...     print("✗ Fix these errors:")
    ...     for error in report['errors']:
    ...         print(f"  - {error}")
    >>> 
    >>> # Apply recommendations
    >>> if report['recommendations']:
    ...     print("Recommendations:")
    ...     for rec in report['recommendations']:
    ...         print(f"  - {rec}")
    
    Notes
    -----
    This function generates a preview image saved as 'video_preview.png'
    showing 3 representative frames from the video.
    """
    report = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'info': {},
        'recommendations': []
    }
    
    # Calculate dt
    times = stack.time()
    dt = np.mean(np.diff(times[1:]))
    
    # Handle frame range
    if end is None or end <= 0:
        end = len(stack)
    end = min(end, len(stack))
    
    n_frames = (end - start) // skip
    duration = n_frames / fps
    
    print("\n" + "="*70)
    print("VIDEO SETTINGS VALIDATION")
    print("="*70)
    
    # ========================================================================
    # 1. CHECK PREPROCESSING
    # ========================================================================
    print("\n1. Checking image preprocessing...")
    try:
        test_frame = preprocessor(stack[0].data.astype(float))
        
        # Handle CuPy arrays
        try:
            if hasattr(test_frame, 'get'):
                test_frame = test_frame.get()
        except:
            pass
        
        # Check statistics
        mean_val = test_frame.mean()
        std_val = test_frame.std()
        
        report['info']['preprocessed_mean'] = mean_val
        report['info']['preprocessed_std'] = std_val
        
        print(f"   Mean intensity: {mean_val:.3f}")
        print(f"   Std deviation: {std_val:.3f}")
        
        # Check for issues
        if mean_val < 0.1:
            report['warnings'].append("Image is very dark (mean < 0.1)")
            report['recommendations'].append(
                "Adjust preprocessing to increase brightness:\n"
                "    img = img - img.min()  # Remove dark offset\n"
                "    img = img / np.percentile(img, 99)"
            )
            print("   ⚠ Image is very dark")
        elif mean_val > 0.9:
            report['warnings'].append("Image is very bright (mean > 0.9)")
            report['warnings'].append("May lose contrast")
            print("   ⚠ Image is very bright")
        
        if std_val < 0.05:
            report['warnings'].append("Very low contrast (std < 0.05)")
            report['recommendations'].append("Check preprocessing - image may appear washed out")
            print("   ⚠ Low contrast")
        
        if 0.2 < mean_val < 0.8 and std_val > 0.1:
            print("   ✓ Preprocessing looks good")
    
    except Exception as e:
        report['errors'].append(f"Preprocessing failed: {e}")
        report['is_valid'] = False
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # 2. CHECK VELOCITY FIELD
    # ========================================================================
    print("\n2. Checking velocity field...")
    try:
        is_timeseries = u.ndim == 3
        
        if is_timeseries:
            u_test = u[0]
            v_test = v[0]
        else:
            u_test = u
            v_test = v
        
        # Calculate statistics
        speed = np.sqrt(u_test**2 + v_test**2)
        speed_valid = speed[np.isfinite(speed) & (speed > 0)]
        
        if len(speed_valid) == 0:
            report['errors'].append("No valid velocity vectors found!")
            report['is_valid'] = False
            print("   ✗ No valid velocity data")
        else:
            speed_mean = speed_valid.mean()
            speed_max = speed_valid.max()
            speed_95 = np.percentile(speed_valid, 95)
            
            # Convert to pixels
            speed_pix = (speed_95 * dt) / pixel_size
            
            report['info']['velocity_mean'] = speed_mean
            report['info']['velocity_max'] = speed_max
            report['info']['velocity_95th'] = speed_95
            report['info']['displacement_pixels'] = speed_pix
            
            print(f"   Mean speed: {speed_mean:.2f} μm/s")
            print(f"   Max speed: {speed_max:.2f} μm/s")
            print(f"   95th percentile: {speed_95:.2f} μm/s")
            print(f"   Displacement (95th): {speed_pix:.1f} pixels")
            
            # Check arrow visibility
            if arrow_scale is None:
                auto_scale = 20.0 / speed_pix if speed_pix > 0 else 1.0
                auto_scale = np.clip(auto_scale, 0.5, 10.0)
            else:
                auto_scale = arrow_scale
            
            arrow_length = speed_pix * auto_scale
            
            if arrow_length < 5:
                report['warnings'].append("Arrows will be very small (< 5 pixels)")
                report['recommendations'].append(
                    f"Increase arrow_scale from {auto_scale:.2f} to {auto_scale * 2:.2f}"
                )
                print(f"   ⚠ Arrows will be small ({arrow_length:.1f} px)")
            elif arrow_length > 100:
                report['warnings'].append("Arrows will be very large (> 100 pixels)")
                report['recommendations'].append(
                    f"Decrease arrow_scale from {auto_scale:.2f} to {auto_scale * 0.5:.2f}"
                )
                print(f"   ⚠ Arrows will be large ({arrow_length:.1f} px)")
            else:
                print(f"   ✓ Arrow length: {arrow_length:.1f} pixels (good)")
            
            # Check coverage
            coverage = (speed > 0).sum() / speed.size
            print(f"   Vector coverage: {coverage*100:.1f}%")
            
            if coverage < 0.1:
                report['warnings'].append(
                    f"Very sparse velocity field ({coverage*100:.1f}% coverage)"
                )
    
    except Exception as e:
        report['errors'].append(f"Velocity field check failed: {e}")
        report['is_valid'] = False
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # 3. CHECK RESOLUTION AND FILE SIZE
    # ========================================================================
    print("\n3. Checking resolution and file size...")
    try:
        original_height, original_width = test_frame.shape
        
        if output_resolution:
            target_width, target_height = output_resolution
        else:
            target_width, target_height = original_width, original_height
        
        # Ensure even dimensions
        target_width = (target_width // 2) * 2
        target_height = (target_height // 2) * 2
        
        print(f"   Original: {original_width}×{original_height}")
        print(f"   Output: {target_width}×{target_height}")
        print(f"   Frames: {n_frames} ({duration:.1f} seconds @ {fps} fps)")
        
        # Estimate file size
        from .utils import estimate_file_size
        estimated_size = estimate_file_size(
            target_width, target_height, n_frames, fps, codec
        )
        
        report['info']['estimated_size_mb'] = estimated_size
        report['info']['target_size_mb'] = target_size_mb
        
        print(f"   Estimated size: {estimated_size:.1f} MB")
        print(f"   Target size: {target_size_mb:.1f} MB")
        
        size_ratio = estimated_size / target_size_mb
        
        if size_ratio > 1.2:
            report['warnings'].append(
                f"Estimated size ({estimated_size:.1f} MB) exceeds target ({target_size_mb:.1f} MB)"
            )
            report['recommendations'].append(
                "Options to reduce size:\n"
                "    - Lower resolution_scale\n"
                "    - Increase skip parameter\n"
                "    - Use codec='h265' for better compression"
            )
            print(f"   ⚠ Will exceed target by {estimated_size - target_size_mb:.1f} MB")
        elif size_ratio < 0.5:
            report['warnings'].append(
                f"File will be much smaller than target ({estimated_size:.1f} vs {target_size_mb:.1f} MB)"
            )
            report['recommendations'].append(
                "Could improve quality:\n"
                "    - Increase resolution\n"
                "    - Decrease skip parameter"
            )
            print(f"   ℹ File will be {target_size_mb - estimated_size:.1f} MB smaller than target")
        else:
            print(f"   ✓ Size estimate within target")
    
    except Exception as e:
        report['errors'].append(f"Resolution check failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # 4. GENERATE PREVIEW
    # ========================================================================
    print("\n4. Generating preview frames...")
    try:
        # Select processor
        if use_gpu:
            from .overlay import GPUVelocityOverlayProcessor as ProcessorClass
        else:
            from .overlay import VelocityOverlayProcessor as ProcessorClass
        
        processor = ProcessorClass(
            preprocessor, x, y, u, v, pixel_size, dt,
            output_resolution=output_resolution if output_resolution else None,
            arrow_scale=arrow_scale,
            arrow_width=arrow_width,
            arrow_color=arrow_color,
            subsample=subsample
        )
        
        # Preview 3 frames
        test_indices = [
            start,
            start + n_frames // 2,
            min(start + n_frames - 1, end - 1)
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        all_black = True
        for idx, frame_idx in enumerate(test_indices):
            img = stack[frame_idx].data.astype(float)
            preview = processor(img, frame_idx)
            
            axes[idx].imshow(preview)
            axes[idx].set_title(f'Frame {frame_idx}\nt = {times[frame_idx]:.2f}s')
            axes[idx].axis('off')
            
            # Check if all black
            if preview.max() > 10:
                all_black = False
            else:
                report['errors'].append(f"Frame {frame_idx} is all black!")
                report['is_valid'] = False
        
        if all_black:
            report['errors'].append("All preview frames are black!")
            report['recommendations'].append(
                "Check preprocessing - ensure output is in [0, 1] or [0, 255] range"
            )
        
        plt.tight_layout()
        preview_path = 'video_preview.png'
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Preview saved to: {preview_path}")
        plt.show()
    
    except Exception as e:
        report['errors'].append(f"Preview generation failed: {e}")
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if report['errors']:
        print("\n✗ ERRORS (must fix):")
        for error in report['errors']:
            print(f"  • {error}")
        report['is_valid'] = False
    
    if report['warnings']:
        print("\n⚠ WARNINGS:")
        for warning in report['warnings']:
            print(f"  • {warning}")
    
    if report['recommendations']:
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    if report['is_valid'] and not report['warnings']:
        print("\n✓ ALL CHECKS PASSED - Ready to create video!")
    elif report['is_valid']:
        print("\n✓ Settings are valid but could be optimized")
    else:
        print("\n✗ VALIDATION FAILED - Please fix errors before proceeding")
    
    print("="*70 + "\n")
    
    return report


def interactive_preview(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_resolution: Optional[Tuple[int, int]] = None,
    use_gpu: bool = False
) -> Dict:
    """
    Interactive preview with adjustable parameters.
    
    Opens a matplotlib window with sliders to adjust:
    - Frame index (see different time points)
    - Arrow scale (adjust arrow length)
    - Subsample (adjust arrow density)
    
    Parameters
    ----------
    stack : object
        Image stack
    preprocessor : callable
        Preprocessing function
    x, y, u, v : ndarray
        Velocity data
    pixel_size : float
        Microns per pixel
    output_resolution : tuple, optional
        Target resolution
    use_gpu : bool, optional
        Use GPU processing
    
    Returns
    -------
    dict
        Optimal parameters: {'arrow_scale': float, 'subsample': int}
    
    Examples
    --------
    >>> # Interactively adjust parameters
    >>> params = interactive_preview(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65
    ... )
    >>> # Use sliders to find best settings, then close window
    >>> print(f"Optimal arrow_scale: {params['arrow_scale']:.2f}")
    >>> print(f"Optimal subsample: {params['subsample']}")
    >>> 
    >>> # Use optimal parameters in video creation
    >>> create_velocity_video(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     arrow_scale=params['arrow_scale'],
    ...     subsample=params['subsample'],
    ...     output_path='bacteria.mp4'
    ... )
    
    Notes
    -----
    Close the preview window when satisfied with the settings.
    The function returns the final slider values.
    """
    # Calculate dt
    times = stack.time()
    dt = np.mean(np.diff(times[1:]))
    
    # Initial parameters
    initial_arrow_scale = 2.0
    initial_subsample = 2
    initial_frame = 0
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    
    # Main image axis
    ax_img = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # Slider axes
    ax_frame = plt.axes([0.15, 0.20, 0.7, 0.03])
    ax_scale = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_subsample = plt.axes([0.15, 0.10, 0.7, 0.03])
    
    # Select processor
    if use_gpu:
        from .overlay import GPUVelocityOverlayProcessor as ProcessorClass
    else:
        from .overlay import VelocityOverlayProcessor as ProcessorClass
    
    # State storage
    state = {'processor': None}
    
    def update_preview(frame_idx, arrow_scale, subsample):
        """Update preview with new parameters."""
        # Create new processor with updated parameters
        state['processor'] = ProcessorClass(
            preprocessor, x, y, u, v, pixel_size, dt,
            output_resolution=output_resolution,
            arrow_scale=arrow_scale,
            subsample=int(subsample)
        )
        
        img = stack[int(frame_idx)].data.astype(float)
        preview = state['processor'](img, int(frame_idx))
        
        ax_img.clear()
        ax_img.imshow(preview)
        ax_img.set_title(
            f'Frame {int(frame_idx)} (t={times[int(frame_idx)]:.2f}s) | '
            f'Arrow scale: {arrow_scale:.2f} | Subsample: {int(subsample)}'
        )
        ax_img.axis('off')
        fig.canvas.draw_idle()
    
    # Create sliders
    slider_frame = Slider(
        ax_frame, 'Frame', 0, len(stack)-1,
        valinit=initial_frame, valstep=1
    )
    slider_scale = Slider(
        ax_scale, 'Arrow Scale', 0.1, 10.0,
        valinit=initial_arrow_scale
    )
    slider_subsample = Slider(
        ax_subsample, 'Subsample', 1, 10,
        valinit=initial_subsample, valstep=1
    )
    
    # Update function
    def update(val):
        update_preview(
            slider_frame.val,
            slider_scale.val,
            slider_subsample.val
        )
    
    slider_frame.on_changed(update)
    slider_scale.on_changed(update)
    slider_subsample.on_changed(update)
    
    # Initial display
    update_preview(initial_frame, initial_arrow_scale, initial_subsample)
    
    print("\n" + "="*70)
    print("INTERACTIVE PREVIEW")
    print("="*70)
    print("Use sliders to adjust parameters:")
    print("  - Frame: Navigate through video")
    print("  - Arrow Scale: Adjust arrow length")
    print("  - Subsample: Adjust arrow density")
    print("\nClose window when satisfied with settings")
    print("="*70 + "\n")
    
    plt.show()
    
    # Return final values
    result = {
        'arrow_scale': slider_scale.val,
        'subsample': int(slider_subsample.val)
    }
    
    print(f"\n✓ Optimal parameters:")
    print(f"  arrow_scale = {result['arrow_scale']:.2f}")
    print(f"  subsample = {result['subsample']}")
    
    return result