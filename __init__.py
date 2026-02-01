"""
Video Creation Package for Microscopy Data Analysis
===================================================

Comprehensive video generation from microscopy image stacks with support for:
- High-quality encoding with multiple codec options
- Grayscale and RGB video creation
- Vector field overlays (PIV visualization)
- GPU-accelerated processing
- FFmpeg integration
- Automatic parameter optimization

Author: Carlo
Version: 2.0.0
"""

__version__ = '2.0.0'
__author__ = 'Carlo'

# Import core video creation functions
from .video_writer import (
    VideoWriter,
    check_available_codecs
)

from .video_basic import (
    video_from_stack,
    video_from_stack_color,
    video_cv2  # Backward compatibility
)

from .video_vectors import (
    video_with_vectors,
    draw_vectors_opencv,
    draw_vectors_gpu
)

from .video_utils import (
    calculate_fps,
    normalize_image,
    add_text_overlay
)

# Import FFmpeg-based functions
from .video_ffmpeg import (
    RobustFFmpegWriter,
    create_velocity_video_robust,
    check_ffmpeg_codecs,
    validate_ffmpeg_params
)

# Import high-level functions
from .video_creator import (
    create_velocity_video,
    create_video_simple,
    create_video_with_validation
)

# Import preview functions
from .preview import (
    preview_single_frame,
    preview_multiple_frames,
    validate_video_settings,
    interactive_preview
)

# Import overlay processors
from .overlay import (
    VelocityOverlayProcessor,
    GPUVelocityOverlayProcessor
)

# Import utilities
from .utils import (
    calculate_optimal_resolution,
    auto_calibrate_arrows,
    mm2inch
)

# Define what's available with "from _video import *"
__all__ = [
    # Video writers
    'VideoWriter',
    'RobustFFmpegWriter',
    
    # Basic video creation
    'video_from_stack',
    'video_from_stack_color',
    'video_cv2',  # Backward compat
    
    # Vector field videos
    'video_with_vectors',
    'draw_vectors_opencv',
    'draw_vectors_gpu',
    
    # High-level creation functions
    'create_velocity_video',
    'create_video_simple',
    'create_video_with_validation',
    'create_velocity_video_robust',
    
    # Preview functions
    'preview_single_frame',
    'preview_multiple_frames',
    'validate_video_settings',
    'interactive_preview',
    
    # Processors
    'VelocityOverlayProcessor',
    'GPUVelocityOverlayProcessor',
    
    # Utilities
    'calculate_fps',
    'normalize_image',
    'add_text_overlay',
    'check_available_codecs',
    'check_ffmpeg_codecs',
    'validate_ffmpeg_params',
    'calculate_optimal_resolution',
    'auto_calibrate_arrows',
    'mm2inch',
]

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Print import status
if GPU_AVAILABLE:
    print(f"_video v{__version__} loaded (GPU acceleration enabled)")
else:
    print(f"_video v{__version__} loaded (CPU only)")

# Convenience function to show all available functions
def list_functions():
    """Print all available video creation functions."""
    print("\n" + "="*70)
    print("Available Video Creation Functions")
    print("="*70)
    print("\nBasic Video Creation:")
    print("  - video_from_stack: Create grayscale video from stack")
    print("  - video_from_stack_color: Create RGB video from stack")
    print("  - create_video_simple: High-level simple video creation")
    
    print("\nVector Field Videos:")
    print("  - video_with_vectors: Create video with velocity field overlay")
    print("  - create_velocity_video: GPU-accelerated velocity video")
    print("  - create_velocity_video_robust: FFmpeg-based robust creation")
    
    print("\nPreview & Validation:")
    print("  - preview_single_frame: Preview one frame with arrows")
    print("  - preview_multiple_frames: Preview multiple frames")
    print("  - validate_video_settings: Check parameters before rendering")
    print("  - interactive_preview: Interactive parameter adjustment")
    
    print("\nUtilities:")
    print("  - calculate_fps: Calculate frame rate from timing")
    print("  - check_available_codecs: Test available video codecs")
    print("  - auto_calibrate_arrows: Automatically calibrate arrow parameters")
    print("="*70 + "\n")
