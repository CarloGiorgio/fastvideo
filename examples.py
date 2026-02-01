"""
FastVideo Usage Examples
======================

Complete working examples demonstrating all major features.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# =============================================================================
# EXAMPLE 1: Basic Video Creation
# =============================================================================

def example_basic_video(stack):
    """
    Create a simple grayscale video from a stack.
    
    Parameters
    ----------
    stack : object
        Image stack with .data and .time() methods
    """
    from fastvideo import video_from_stack
    
    # Define simple normalization preprocessing
    def preprocess(img):
        # Normalize to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min())
        return img_norm
    
    # Create video with default settings
    video_from_stack(
        stack,
        preprocess,
        'output_basic.mp4',
        speed=1.0,
        codec='h264',
        quality='medium'
    )
    
    print("✓ Basic video created: output_basic.mp4")


# =============================================================================
# EXAMPLE 2: High-Quality Video with Text Overlay
# =============================================================================

def example_high_quality_with_text(stack):
    """Create high-quality video with time overlay."""
    from fastvideo import video_from_stack
    
    def preprocess(img):
        return img / img.max()
    
    video_from_stack(
        stack,
        preprocess,
        'output_hq_text.mp4',
        speed=2.0,           # 2x playback speed
        codec='h264',
        quality='high',      # High quality
        text=True,           # Show time overlay
        fontsize=1.5,        # Moderate font size
        text_color=(255, 255, 0),  # Yellow text
        text_position=(50, 50)     # Top-left corner
    )
    
    print("✓ High-quality video with text created")


# =============================================================================
# EXAMPLE 3: Advanced Preprocessing
# =============================================================================

def example_advanced_preprocessing(stack):
    """Create video with sophisticated image processing."""
    from fastvideo import video_from_stack
    
    def advanced_preprocess(img):
        """
        Advanced preprocessing pipeline:
        1. Gaussian smoothing
        2. Background subtraction
        3. Normalization
        """
        # Apply Gaussian smoothing
        img_smooth = gaussian_filter(img, sigma=1.5)
        
        # Estimate and subtract background
        background = gaussian_filter(img_smooth, sigma=50)
        img_subtracted = img_smooth - background
        
        # Clip negative values
        img_subtracted = np.maximum(img_subtracted, 0)
        
        # Normalize to [0, 1]
        if img_subtracted.max() > 0:
            img_norm = img_subtracted / img_subtracted.max()
        else:
            img_norm = img_subtracted
        
        return img_norm
    
    video_from_stack(
        stack,
        advanced_preprocess,
        'output_processed.mp4',
        codec='h264',
        quality='medium'
    )
    
    print("✓ Processed video created: output_processed.mp4")


# =============================================================================
# EXAMPLE 4: Velocity Field Video (PIV)
# =============================================================================

def example_velocity_video(stack, x, y, u, v, pixel_size=0.65):
    """
    Create video with velocity field overlay.
    
    Parameters
    ----------
    stack : object
        Image stack
    x, y : np.ndarray
        Coordinate meshgrids
    u, v : np.ndarray
        Velocity components (can be 2D or 3D)
    pixel_size : float
        Physical size per pixel (microns)
    """
    from fastvideo import create_velocity_video
    
    def preprocess(img):
        return (img - img.min()) / (img.max() - img.min())
    
    create_velocity_video(
        stack, preprocess, x, y, u, v,
        filename='velocity_video.mp4',
        pixel_size=pixel_size,
        vector_scale=None,         # Auto-calibrate
        vector_skip=2,             # Draw every 2nd arrow
        vector_color=(255, 255, 0),  # Yellow arrows
        vector_thickness=2,
        quality='high',
        text=True                  # Show timestamps
    )
    
    print("✓ Velocity video created: velocity_video.mp4")


# =============================================================================
# EXAMPLE 5: Manual Arrow Scaling
# =============================================================================

def example_manual_arrows(stack, x, y, u, v):
    """
    Create velocity video with manual arrow parameters.
    """
    from fastvideo import video_with_vectors, calculate_auto_arrow_scale
    
    def preprocess(img):
        return img / img.max()
    
    # Calculate recommended scale
    auto_scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)
    print(f"Auto-calculated scale: {auto_scale:.2f}")
    
    # Use custom scale (e.g., make arrows longer)
    custom_scale = auto_scale * 1.5
    
    video_with_vectors(
        stack, preprocess, x, y, u, v,
        filename='velocity_custom.mp4',
        vector_scale=custom_scale,
        vector_skip=3,                # Fewer arrows for clarity
        vector_color=(0, 255, 0),     # Green arrows
        vector_thickness=3,           # Thicker arrows
        codec='h264',
        quality='high'
    )
    
    print(f"✓ Custom arrow video created (scale={custom_scale:.2f})")


# =============================================================================
# EXAMPLE 6: Frame Subset Processing
# =============================================================================

def example_frame_subset(stack):
    """Process only a subset of frames."""
    from fastvideo import video_from_stack
    
    def preprocess(img):
        return img / img.max()
    
    # Get total frames
    n_frames = len(stack)
    print(f"Total frames in stack: {n_frames}")
    
    # Process middle third, using every 2nd frame
    start = n_frames // 3
    end = 2 * n_frames // 3
    
    video_from_stack(
        stack,
        preprocess,
        'subset_video.mp4',
        start=start,
        end=end,
        skip=2,              # Use every 2nd frame
        speed=1.0,
        quality='medium'
    )
    
    print(f"✓ Subset video created (frames {start}-{end}, skip=2)")


# =============================================================================
# EXAMPLE 7: RGB Color Video
# =============================================================================

def example_color_video(stack):
    """Create RGB color video using matplotlib colormap."""
    from fastvideo import video_from_stack_color
    import matplotlib.pyplot as plt
    
    def rgb_preprocess(img):
        """Convert grayscale to RGB using viridis colormap."""
        # Normalize to [0, 1]
        img_norm = (img - img.min()) / (img.max() - img.min())
        
        # Apply colormap (returns RGBA)
        rgba = plt.cm.viridis(img_norm)
        
        # Extract RGB only
        rgb = rgba[:, :, :3]
        
        # Convert to uint8
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        return rgb_uint8
    
    video_from_stack_color(
        stack,
        rgb_preprocess,
        'color_video.mp4',
        codec='h264',
        quality='high'
    )
    
    print("✓ Color video created: color_video.mp4")


# =============================================================================
# EXAMPLE 8: Multiple Quality Versions
# =============================================================================

def example_multiple_qualities(stack):
    """Create same video in different quality settings."""
    from fastvideo import video_from_stack
    
    def preprocess(img):
        return img / img.max()
    
    qualities = ['low', 'medium', 'high']
    
    for quality in qualities:
        filename = f'output_{quality}.mp4'
        
        video_from_stack(
            stack,
            preprocess,
            filename,
            speed=2.0,
            codec='h264',
            quality=quality,
            verbose=False  # Suppress progress bars for clean output
        )
        
        # Get file size
        import os
        size_mb = os.path.getsize(filename) / (1024**2)
        print(f"✓ {quality.capitalize()}: {filename} ({size_mb:.1f} MB)")


# =============================================================================
# EXAMPLE 9: Check System Capabilities
# =============================================================================

def example_check_system():
    """Check available codecs and system information."""
    from fastvideo import check_available_codecs, get_system_info
    
    print("\n" + "="*60)
    print("SYSTEM CAPABILITIES CHECK")
    print("="*60)
    
    # Check available codecs
    print("\nAvailable codecs:")
    codecs = check_available_codecs(verbose=True)
    
    # Get system info
    print("\nSystem information:")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("="*60 + "\n")


# =============================================================================
# EXAMPLE 10: Codec Comparison
# =============================================================================

def example_codec_comparison(stack):
    """Compare different codecs for the same video."""
    from fastvideo import video_from_stack
    import os
    
    def preprocess(img):
        return img / img.max()
    
    codecs = ['h264', 'h265', 'mp4v']
    
    print("\nCodec Comparison:")
    print("-" * 60)
    
    for codec in codecs:
        filename = f'output_{codec}.mp4'
        
        try:
            video_from_stack(
                stack,
                preprocess,
                filename,
                speed=2.0,
                codec=codec,
                quality='medium',
                start=0,
                end=100,  # Just first 100 frames for comparison
                verbose=False
            )
            
            # Get file size
            size_mb = os.path.getsize(filename) / (1024**2)
            print(f"✓ {codec.upper()}: {size_mb:.1f} MB")
            
        except Exception as e:
            print(f"✗ {codec.upper()}: Failed ({str(e)[:50]}...)")
    
    print("-" * 60)


# =============================================================================
# EXAMPLE 11: Complete PIV Workflow
# =============================================================================

def example_complete_piv_workflow(stack, piv_results, pixel_size=0.65):
    """
    Complete workflow: preview → auto-calibrate → create video.
    
    Parameters
    ----------
    stack : object
        Image stack
    piv_results : dict
        Dictionary with 'x', 'y', 'u', 'v' arrays
    pixel_size : float
        Pixel size in microns
    """
    from fastvideo import calculate_auto_arrow_scale, create_velocity_video
    
    x = piv_results['x']
    y = piv_results['y']
    u = piv_results['u']
    v = piv_results['v']
    
    def preprocess(img):
        return (img - img.min()) / (img.max() - img.min())
    
    # Step 1: Calculate optimal arrow scale
    print("Step 1: Calculating optimal arrow scale...")
    arrow_scale = calculate_auto_arrow_scale(u, v, pixel_size)
    print(f"  → Recommended scale: {arrow_scale:.2f}")
    
    # Step 2: Calculate velocity statistics
    print("\nStep 2: Analyzing velocity field...")
    velocity_mag = np.sqrt(u**2 + v**2)
    print(f"  → Median velocity: {np.median(velocity_mag):.2f} μm/s")
    print(f"  → Max velocity: {np.max(velocity_mag):.2f} μm/s")
    
    # Step 3: Create final video
    print("\nStep 3: Creating video...")
    create_velocity_video(
        stack, preprocess, x, y, u, v,
        filename='piv_final.mp4',
        pixel_size=pixel_size,
        vector_scale=arrow_scale,
        vector_skip=2,
        vector_color=(255, 255, 0),
        quality='high',
        text=True,
        fontsize=1.5
    )
    
    print("✓ Complete PIV workflow finished!")


# =============================================================================
# EXAMPLE 12: Error Handling
# =============================================================================

def example_error_handling(stack):
    """Demonstrate proper error handling."""
    from fastvideo import video_from_stack
    
    def preprocess(img):
        return img / img.max()
    
    try:
        # This might fail if codec not available
        video_from_stack(
            stack,
            preprocess,
            'test_video.mp4',
            codec='h265',  # Might not be available
            quality='high'
        )
        print("✓ Video created successfully")
        
    except RuntimeError as e:
        print(f"✗ Video creation failed: {e}")
        print("→ Trying fallback codec...")
        
        # Try with more compatible codec
        video_from_stack(
            stack,
            preprocess,
            'test_video.mp4',
            codec='h264',  # More widely available
            quality='high'
        )
        print("✓ Video created with fallback codec")
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_all_examples(stack=None, piv_data=None):
    """
    Run all examples (requires valid stack object).
    
    Parameters
    ----------
    stack : object, optional
        Image stack. If None, examples are just demonstrated.
    piv_data : dict, optional
        PIV results with 'x', 'y', 'u', 'v'
    """
    if stack is None:
        print("No stack provided - showing example code only")
        print("To run examples, provide a valid image stack object")
        return
    
    print("\n" + "="*70)
    print("FastVideo - Running All Examples")
    print("="*70 + "\n")
    
    example_check_system()
    example_basic_video(stack)
    example_high_quality_with_text(stack)
    example_advanced_preprocessing(stack)
    example_frame_subset(stack)
    example_multiple_qualities(stack)
    example_codec_comparison(stack)
    
    if piv_data is not None:
        example_velocity_video(
            stack, 
            piv_data['x'], piv_data['y'],
            piv_data['u'], piv_data['v']
        )
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    # If run as script, show available examples
    import fastvideo
    print(f"\nFastVideo v{fastvideo.__version__} - Examples Module")
    print("="*70)
    print("\nAvailable example functions:")
    print("  - example_basic_video(stack)")
    print("  - example_high_quality_with_text(stack)")
    print("  - example_advanced_preprocessing(stack)")
    print("  - example_velocity_video(stack, x, y, u, v)")
    print("  - example_manual_arrows(stack, x, y, u, v)")
    print("  - example_frame_subset(stack)")
    print("  - example_color_video(stack)")
    print("  - example_multiple_qualities(stack)")
    print("  - example_check_system()")
    print("  - example_codec_comparison(stack)")
    print("  - example_complete_piv_workflow(stack, piv_results)")
    print("  - example_error_handling(stack)")
    print("\nTo run all: run_all_examples(stack, piv_data)")
    print("="*70)
