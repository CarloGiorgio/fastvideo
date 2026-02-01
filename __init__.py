"""
FastVideo - Scientific Microscopy Video Creation Package
======================================================

A comprehensive, production-ready package for creating high-quality videos from 
microscopy image stacks with optional velocity field overlays.

Features
--------
- **Multiple codec support**: H264, H265, XVID, MJPEG
- **GPU acceleration**: CUDA-accelerated processing when available
- **Quality control**: Automatic parameter optimization and validation
- **Vector field overlays**: PIV visualization with auto-calibration
- **Preview system**: Interactive parameter tuning before rendering
- **Batch processing**: Parallel video creation for multiple datasets
- **FFmpeg integration**: Hardware encoding support (NVENC)
- **Robust error handling**: Graceful fallbacks and comprehensive diagnostics

Quick Start
-----------
Basic video creation:

    >>> from fastvideo import video_from_stack
    >>> 
    >>> # Define preprocessing
    >>> def preprocess(img):
    ...     return (img - img.min()) / (img.max() - img.min())
    >>> 
    >>> # Create video
    >>> video_from_stack(stack, preprocess, 'output.mp4',
    ...                  speed=2.0, codec='h264', quality='high')

Velocity field video:

    >>> from fastvideo import create_velocity_video
    >>> 
    >>> create_velocity_video(
    ...     stack, preprocess, x, y, u, v,
    ...     filename='velocity.mp4',
    ...     pixel_size=0.65,
    ...     arrow_scale=None  # Auto-calibrate
    ... )

Interactive preview:

    >>> from fastvideo import interactive_preview
    >>> 
    >>> interactive_preview(stack, preprocess, x, y, u, v,
    ...                     pixel_size=0.65)

Author: Carlo
Version: 2.0.0
License: MIT
"""

__version__ = '2.0.0'
__author__ = 'Carlo'
__license__ = 'MIT'

# =============================================================================
# IMPORTS - Core Functionality
# =============================================================================

# Video writers and core utilities
from .video_writer import (
    VideoWriter,
    check_available_codecs
)

from .video_utils import (
    calculate_fps,
    normalize_image,
    add_text_overlay,
    validate_frame_range
)

# Basic video creation
from .video_basic import (
    video_from_stack,
    video_from_stack_color
)

# Vector field visualization
from .video_vectors import (
    video_with_vectors,
    draw_vectors_opencv,
    draw_vectors_gpu,
    calculate_auto_arrow_scale
)

# High-level convenience functions
from .video_creator import (
    create_video_simple,
    create_velocity_video,
    create_video_with_validation
)

# Preview and validation
from .preview import (
    preview_single_frame,
    preview_multiple_frames,
    validate_video_settings,
    interactive_preview
)

# Batch processing
from .batch import (
    VideoConfig,
    ParallelVideoBatch,
    make_batch_videos
)

# FFmpeg integration
from .ffmpeg_integration import (
    RobustFFmpegWriter,
    create_velocity_video_ffmpeg,
    check_ffmpeg_codecs,
    validate_ffmpeg_params
)

# Overlay processors
from .overlay import (
    VelocityOverlayProcessor,
    GPUVelocityOverlayProcessor
)

# Utilities
from .utils import (
    calculate_optimal_resolution,
    auto_calibrate_arrows,
    mm2inch
)

# =============================================================================
# PUBLIC API - What users can import
# =============================================================================

__all__ = [
    # === Core Video Creation ===
    'video_from_stack',           # Grayscale video
    'video_from_stack_color',     # RGB video
    'video_with_vectors',         # Video with velocity overlay
    
    # === High-Level Functions ===
    'create_video_simple',        # Quick video with defaults
    'create_velocity_video',      # Velocity video with auto-calibration
    'create_video_with_validation',  # Video with preview validation
    'create_velocity_video_ffmpeg',  # FFmpeg-based velocity video
    
    # === Preview & Validation ===
    'preview_single_frame',       # Preview one frame
    'preview_multiple_frames',    # Preview multiple frames
    'validate_video_settings',    # Comprehensive validation
    'interactive_preview',        # Interactive parameter tuning
    
    # === Batch Processing ===
    'VideoConfig',                # Configuration dataclass
    'ParallelVideoBatch',         # Parallel batch processor
    'make_batch_videos',          # Convenience batch function
    
    # === Utilities ===
    'VideoWriter',                # Low-level video writer
    'RobustFFmpegWriter',         # FFmpeg writer
    'VelocityOverlayProcessor',   # CPU overlay processor
    'GPUVelocityOverlayProcessor',  # GPU overlay processor
    'calculate_fps',              # FPS calculation
    'normalize_image',            # Image normalization
    'add_text_overlay',           # Text overlay
    'validate_frame_range',       # Frame range validation
    'draw_vectors_opencv',        # Draw vectors on image
    'draw_vectors_gpu',           # GPU vector drawing
    'calculate_auto_arrow_scale', # Auto arrow scale
    'check_available_codecs',     # Test codecs
    'check_ffmpeg_codecs',        # Test FFmpeg codecs
    'validate_ffmpeg_params',     # Validate FFmpeg params
    'calculate_optimal_resolution',  # Resolution optimization
    'auto_calibrate_arrows',      # Arrow calibration
    'mm2inch',                    # Unit conversion
]

# =============================================================================
# SYSTEM STATUS
# =============================================================================

# Check for optional dependencies
try:
    import cupy as cp
    GPU_AVAILABLE = True
    _gpu_status = "enabled"
except ImportError:
    GPU_AVAILABLE = False
    _gpu_status = "disabled"

try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, text=True, timeout=2)
    FFMPEG_AVAILABLE = result.returncode == 0
    _ffmpeg_status = "available"
except (FileNotFoundError, subprocess.TimeoutExpired):
    FFMPEG_AVAILABLE = False
    _ffmpeg_status = "not found"

# Print status on import
print(f"FastVideo v{__version__} loaded")
print(f"  GPU acceleration: {_gpu_status}")
print(f"  FFmpeg support: {_ffmpeg_status}")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def list_functions():
    """
    Print all available functions organized by category.
    
    Examples
    --------
    >>> import fastvideo
    >>> fastvideo.list_functions()
    """
    print("\n" + "="*70)
    print(f"FastVideo v{__version__} - Available Functions")
    print("="*70)
    
    print("\n📹 BASIC VIDEO CREATION")
    print("  • video_from_stack()        - Create grayscale video from stack")
    print("  • video_from_stack_color()  - Create RGB video from stack")
    print("  • create_video_simple()     - Quick video with sensible defaults")
    
    print("\n🎯 VELOCITY FIELD VIDEOS")
    print("  • video_with_vectors()           - Video with velocity overlay")
    print("  • create_velocity_video()        - Auto-calibrated velocity video")
    print("  • create_velocity_video_ffmpeg() - FFmpeg-based high quality")
    
    print("\n👁️  PREVIEW & VALIDATION")
    print("  • preview_single_frame()      - Preview one frame with arrows")
    print("  • preview_multiple_frames()   - Preview multiple frames")
    print("  • validate_video_settings()   - Comprehensive parameter check")
    print("  • interactive_preview()       - Interactive parameter tuning")
    
    print("\n⚡ BATCH PROCESSING")
    print("  • make_batch_videos()    - Process multiple videos in parallel")
    print("  • ParallelVideoBatch()   - Advanced batch control")
    print("  • VideoConfig            - Configuration dataclass")
    
    print("\n🔧 UTILITIES")
    print("  • check_available_codecs()    - Test available video codecs")
    print("  • check_ffmpeg_codecs()       - Test FFmpeg codec support")
    print("  • calculate_fps()             - Calculate frame rate")
    print("  • auto_calibrate_arrows()     - Auto-calibrate arrow display")
    print("  • calculate_optimal_resolution() - Resolution optimization")
    
    print("\n📊 SYSTEM STATUS")
    print(f"  GPU acceleration: {_gpu_status}")
    print(f"  FFmpeg support: {_ffmpeg_status}")
    print(f"  Available codecs: ", end="")
    
    # Quick codec check
    try:
        codecs = check_available_codecs(verbose=False)
        print(", ".join(codecs) if codecs else "none")
    except:
        print("(run check_available_codecs() for details)")
    
    print("\n" + "="*70)
    print("For detailed help: help(fastvideo.function_name)")
    print("="*70 + "\n")


def show_examples():
    """
    Print example usage code for common tasks.
    
    Examples
    --------
    >>> import fastvideo
    >>> fastvideo.show_examples()
    """
    examples = """
=======================================================================
FastVideo - Example Usage
=======================================================================

1. BASIC VIDEO CREATION
------------------------
from fastvideo import video_from_stack

# Define preprocessing
def preprocess(img):
    return (img - img.min()) / (img.max() - img.min())

# Create video
video_from_stack(
    stack, preprocess, 'output.mp4',
    speed=2.0,
    codec='h264',
    quality='high',
    text=True  # Show time overlay
)


2. VELOCITY FIELD VIDEO
-------------------------
from fastvideo import create_velocity_video

create_velocity_video(
    stack, preprocess, x, y, u, v,
    filename='velocity.mp4',
    pixel_size=0.65,
    arrow_scale=None,  # Auto-calibrate
    subsample=2,       # Draw every 2nd arrow
    arrow_color=(255, 255, 0)  # Yellow
)


3. PREVIEW BEFORE RENDERING
----------------------------
from fastvideo import preview_single_frame
import matplotlib.pyplot as plt

preview = preview_single_frame(
    stack, preprocess, x, y, u, v,
    pixel_size=0.65,
    frame_idx=100
)

plt.figure(figsize=(10, 10))
plt.imshow(preview)
plt.title('Preview')
plt.show()


4. VALIDATE PARAMETERS
-----------------------
from fastvideo import validate_video_settings

validate_video_settings(
    stack, preprocess, x, y, u, v,
    pixel_size=0.65,
    output_resolution=(1920, 1080)
)


5. BATCH PROCESSING
--------------------
from fastvideo import make_batch_videos

configs = [
    {'stack': stack1, 'output': 'video1.mp4', 'speed': 1.0},
    {'stack': stack2, 'output': 'video2.mp4', 'speed': 2.0},
    {'stack': stack3, 'output': 'video3.mp4', 'speed': 0.5}
]

make_batch_videos(
    configs,
    preprocess=lambda x: x / x.max(),
    n_workers=4
)


6. CUSTOM PREPROCESSING
------------------------
from scipy.ndimage import gaussian_filter

def custom_preprocess(img):
    # Smooth and background subtract
    smooth = gaussian_filter(img, sigma=1.5)
    background = gaussian_filter(smooth, sigma=50)
    result = smooth - background
    
    # Normalize
    result = (result - result.min()) / (result.max() - result.min())
    return result

video_from_stack(stack, custom_preprocess, 'processed.mp4')


7. CHECK SYSTEM CAPABILITIES
------------------------------
from fastvideo import check_available_codecs, check_ffmpeg_codecs

# Check OpenCV codecs
check_available_codecs()

# Check FFmpeg codecs
check_ffmpeg_codecs()

=======================================================================
"""
    print(examples)


# =============================================================================
# MODULE METADATA
# =============================================================================

def get_version():
    """Return package version string."""
    return __version__


def get_system_info():
    """
    Get comprehensive system information for debugging.
    
    Returns
    -------
    dict
        Dictionary with system capabilities
    
    Examples
    --------
    >>> import fastvideo
    >>> info = fastvideo.get_system_info()
    >>> print(info)
    """
    import sys
    import cv2
    
    info = {
        'fastvideo_version': __version__,
        'python_version': sys.version,
        'opencv_version': cv2.__version__,
        'gpu_available': GPU_AVAILABLE,
        'ffmpeg_available': FFMPEG_AVAILABLE,
    }
    
    if GPU_AVAILABLE:
        try:
            import cupy as cp
            info['cupy_version'] = cp.__version__
            info['cuda_available'] = cp.cuda.is_available()
            if cp.cuda.is_available():
                info['cuda_device'] = cp.cuda.Device().name.decode()
        except:
            pass
    
    try:
        codecs = check_available_codecs(verbose=False)
        info['available_codecs'] = codecs
    except:
        info['available_codecs'] = []
    
    return info
