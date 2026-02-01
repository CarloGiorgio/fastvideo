"""
Video Creation Examples
=======================

Complete working examples for the _video package.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# =============================================================================
# Example 1: Basic Video Creation
# =============================================================================

def example_basic_video():
    """Create a simple grayscale video from a stack."""
    from _video import video_from_stack
    
    # Assume you have a stack object
    # stack = load_stack('data.tif')
    
    # Define preprocessing
    def preprocess(img):
        # Normalize to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min())
        return img_norm
    
    # Create video
    video_from_stack(
        stack,
        preprocess,
        'output_basic.mp4',
        speed=2.0,       # 2x playback speed
        codec='h264',
        quality='high',
        text=True,       # Show time overlay
        start=0,
        end=1000,
        skip=2           # Use every 2nd frame
    )
    
    print("Basic video created: output_basic.mp4")


# =============================================================================
# Example 2: Custom Preprocessing
# =============================================================================

def example_custom_preprocessing():
    """Create video with custom image processing."""
    from _video import video_from_stack
    
    def custom_preprocess(img):
        # Apply Gaussian smoothing
        img_smooth = gaussian_filter(img, sigma=1.5)
        
        # Background subtraction
        background = gaussian_filter(img_smooth, sigma=50)
        img_subtracted = img_smooth - background
        
        # Normalize
        img_norm = (img_subtracted - img_subtracted.min())
        img_norm = img_norm / img_norm.max()
        
        return img_norm
    
    video_from_stack(
        stack,
        custom_preprocess,
        'output_processed.mp4',
        codec='h264',
        quality='medium'
    )
    
    print("Processed video created: output_processed.mp4")


# =============================================================================
# Example 3: Velocity Field Video
# =============================================================================

def example_velocity_video():
    """Create video with velocity field overlay."""
    from _video import video_with_vectors
    
    # Assume you have PIV results
    # x, y = meshgrid of coordinates
    # u, v = velocity components (shape: n_frames x n_rows x n_cols)
    
    def preprocess(img):
        return img / img.max()
    
    video_with_vectors(
        stack, preprocess, x, y, u, v,
        'velocity_field.mp4',
        vector_skip=3,              # Draw every 3rd arrow
        vector_scale=2.0,           # Arrow length scale
        vector_color=(255, 255, 0), # Yellow arrows
        vector_thickness=2,
        text=True,
        fontsize=1.5,
        speed=1.0,
        fps=30
    )
    
    print("Velocity video created: velocity_field.mp4")


# =============================================================================
# Example 4: Auto-Calibrated Velocity Video
# =============================================================================

def example_auto_calibrated_video():
    """Create velocity video with automatic parameter calibration."""
    from _video import create_velocity_video, auto_calibrate_arrows
    
    # Define pixel size (microns per pixel)
    pixel_size = 0.65
    
    # Auto-calibrate arrow parameters
    params = auto_calibrate_arrows(u, v, pixel_size, target_length_pixels=20)
    
    # Create video with calibrated parameters
    def preprocess(img):
        return gaussian_filter(img, sigma=1.0) / img.max()
    
    create_velocity_video(
        stack, preprocess, x, y, u, v,
        'velocity_calibrated.mp4',
        pixel_size=pixel_size,
        vector_scale=params['scale'],
        vector_skip=params['subsample'],
        vector_thickness=params['thickness'],
        codec='h264',
        quality='high'
    )
    
    print(f"Calibrated video created with:")
    print(f"  Arrow scale: {params['scale']:.2f}")
    print(f"  Subsample: {params['subsample']}")
    print(f"  Thickness: {params['thickness']}")


# =============================================================================
# Example 5: Color Video
# =============================================================================

def example_color_video():
    """Create RGB color video."""
    from _video import video_from_stack_color
    import matplotlib.pyplot as plt
    
    def color_preprocess(img, frame_idx):
        # Normalize image
        img_norm = (img - img.min()) / (img.max() - img.min())
        
        # Create colormap (hot colormap effect)
        r = img_norm
        g = img_norm ** 2
        b = img_norm ** 4
        
        # Stack to RGB
        rgb = np.stack([r, g, b], axis=-1)
        
        return rgb
    
    video_from_stack_color(
        stack,
        color_preprocess,
        'output_color.mp4',
        codec='h264',
        quality='high',
        speed=1.5
    )
    
    print("Color video created: output_color.mp4")


# =============================================================================
# Example 6: Resolution Optimization
# =============================================================================

def example_resolution_optimization():
    """Optimize video resolution for target file size."""
    from _video import calculate_optimal_resolution, estimate_file_size
    
    # Original dimensions
    original_size = (1648, 1648)
    
    # Video parameters
    duration = len(stack) / 30  # seconds
    fps = 30
    
    # Estimate file size at original resolution
    size_original = estimate_file_size(
        *original_size, duration, fps, 
        codec='h264', quality='medium'
    )
    print(f"Estimated size at {original_size}: {size_original:.1f} MB")
    
    # Calculate optimal resolution for target size
    target_size_mb = 200
    optimal_size = calculate_optimal_resolution(
        original_size,
        target_size_mb=target_size_mb,
        duration_s=duration,
        fps=fps,
        codec='h264'
    )
    
    print(f"Optimal resolution for {target_size_mb} MB: {optimal_size}")
    
    # Create video at optimized resolution
    def preprocess_scaled(img):
        from scipy.ndimage import zoom
        
        # Scale down to optimal resolution
        scale_y = optimal_size[1] / img.shape[0]
        scale_x = optimal_size[0] / img.shape[1]
        
        img_scaled = zoom(img, (scale_y, scale_x), order=1)
        return img_scaled / img_scaled.max()
    
    from _video import video_from_stack
    video_from_stack(
        stack,
        preprocess_scaled,
        'output_optimized.mp4',
        codec='h264',
        quality='medium',
        fps=fps
    )


# =============================================================================
# Example 7: Batch Processing
# =============================================================================

def example_batch_processing():
    """Process multiple videos with different parameters."""
    from _video import video_from_stack
    
    # Define multiple preprocessing functions
    preprocessors = {
        'raw': lambda img: img / img.max(),
        'smooth': lambda img: gaussian_filter(img, 1.5) / img.max(),
        'sharp': lambda img: img - gaussian_filter(img, 2.0),
    }
    
    # Create videos with each preprocessor
    for name, preprocess in preprocessors.items():
        output_file = f'output_{name}.mp4'
        
        print(f"Creating {output_file}...")
        video_from_stack(
            stack,
            preprocess,
            output_file,
            codec='h264',
            quality='medium',
            speed=2.0,
            skip=2
        )
    
    print("Batch processing complete!")


# =============================================================================
# Example 8: Preview Before Rendering
# =============================================================================

def example_preview():
    """Preview frames before creating full video."""
    from _video import preview_multiple_frames, validate_video_settings
    
    def preprocess(img):
        return gaussian_filter(img, sigma=1.0) / img.max()
    
    # Preview first, middle, and last frames
    print("Previewing frames...")
    preview_multiple_frames(
        stack, 
        preprocess,
        frame_indices=[0, len(stack)//2, len(stack)-1]
    )
    
    # Validate settings
    settings = {
        'filename': 'output.mp4',
        'speed': 2.0,
        'codec': 'h264',
        'quality': 'high',
        'skip': 2
    }
    
    if validate_video_settings(**settings):
        print("Settings validated, proceeding with video creation...")
        from _video import video_from_stack
        video_from_stack(stack, preprocess, **settings)


# =============================================================================
# Example 9: Check Available Codecs
# =============================================================================

def example_check_codecs():
    """Check which video codecs are available on your system."""
    from _video import check_available_codecs
    
    print("Checking available codecs...")
    available = check_available_codecs()
    
    if 'h264' in available:
        print("\nH.264 codec is available - recommended for general use")
    
    if 'h265' in available:
        print("H.265 codec is available - best for high compression")
    
    if not available:
        print("\nWarning: No optimized codecs found.")
        print("Consider installing ffmpeg with codec support.")


# =============================================================================
# Example 10: Complete PIV Workflow
# =============================================================================

def example_complete_piv_workflow():
    """Complete workflow from PIV analysis to video."""
    from _video import (
        auto_calibrate_arrows,
        create_velocity_video,
        estimate_file_size
    )
    
    # Assume you have PIV results
    pixel_size = 0.65  # microns per pixel
    
    # Step 1: Auto-calibrate parameters
    print("Step 1: Calibrating arrow parameters...")
    params = auto_calibrate_arrows(u, v, pixel_size)
    
    # Step 2: Estimate file size
    print("\nStep 2: Estimating file size...")
    duration = len(stack) / 30
    size_est = estimate_file_size(
        stack[0].data.shape[1], stack[0].data.shape[0],
        duration, 30, codec='h264', quality='high'
    )
    print(f"Estimated file size: {size_est:.1f} MB")
    
    # Step 3: Create video
    print("\nStep 3: Creating velocity video...")
    
    def preprocess(img):
        # Smooth and normalize
        img_smooth = gaussian_filter(img, sigma=1.0)
        return img_smooth / img_smooth.max()
    
    create_velocity_video(
        stack, preprocess, x, y, u, v,
        'piv_complete.mp4',
        pixel_size=pixel_size,
        vector_scale=params['scale'],
        vector_skip=params['subsample'],
        vector_color=(0, 255, 255),  # Cyan arrows
        text=True,
        fontsize=1.2,
        codec='h264',
        quality='high',
        fps=30
    )
    
    print("\nComplete PIV workflow finished!")
    print(f"Output: piv_complete.mp4")


# =============================================================================
# Main execution
# =============================================================================

if __name__ == '__main__':
    print("Video Creation Examples")
    print("=" * 70)
    print("\nAvailable examples:")
    print("1. example_basic_video()")
    print("2. example_custom_preprocessing()")
    print("3. example_velocity_video()")
    print("4. example_auto_calibrated_video()")
    print("5. example_color_video()")
    print("6. example_resolution_optimization()")
    print("7. example_batch_processing()")
    print("8. example_preview()")
    print("9. example_check_codecs()")
    print("10. example_complete_piv_workflow()")
    print("\nRun any example function to see it in action.")
    print("\nFirst, check available codecs:")
    example_check_codecs()
