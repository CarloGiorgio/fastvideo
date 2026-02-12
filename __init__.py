"""
SLMicro Video - High-Performance Microscopy Video Creation
===========================================================

Create ultra-compressed videos from microscopy stacks with velocity field overlays.

Features:
- GPU-accelerated processing (CuPy + CUDA)
- Hardware video encoding (NVENC)
- Automatic resolution optimization
- Preview and validation before rendering
- Robust error handling

Author: Carlo
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Carlo'

# Import main functions for easy access
from .preview import (
    preview_single_frame,
    preview_multiple_frames,
    validate_video_settings,
    interactive_preview
)

from .video_creator import (
    create_velocity_video,
    create_video_simple,
    create_video_with_validation
)

from .overlay import (
    VelocityOverlayProcessor,
    GPUVelocityOverlayProcessor
)

from .ffmpeg_writer import (
    RobustFFmpegWriter,
    check_ffmpeg_codecs,
    validate_ffmpeg_params
)

from .utils import (
    calculate_optimal_resolution,
    auto_calibrate_arrows,
    mm2inch
)

# Define what's available with "from slmicro_video import *"
__all__ = [
    # Preview functions
    'preview_single_frame',
    'preview_multiple_frames',
    'validate_video_settings',
    'interactive_preview',
    
    # Main creation functions
    'create_video',
    'create_velocity_video',
    'create_video_simple',
    'create_video_with_validation',
    
    # Processors
    'VelocityOverlayProcessor',
    'GPUVelocityOverlayProcessor',
    
    # FFmpeg utilities
    'RobustFFmpegWriter',
    'check_ffmpeg_codecs',
    'validate_ffmpeg_params',
    
    # Utilities
    'calculate_optimal_resolution',
    'auto_calibrate_arrows',
]

# Print import status
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    print(f"slmicro_video v{__version__} loaded (GPU acceleration enabled)")
else:
    print(f"slmicro_video v{__version__} loaded (CPU only)")