# FastVideo Quick Start Guide

This guide walks you through creating your first microscopy video with velocity overlays in 5 minutes.

## Prerequisites

```bash
# Install core dependencies
pip install numpy scipy matplotlib opencv-python numba

# Install GPU support (optional but recommended)
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12

# Install FFmpeg (system-wide)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

## Basic Workflow

### Step 1: Load Your Data

```python
from cam import loadstack
import numpy as np

# Load microscopy stack
stack = loadstack('bacteria_experiment.zip')

# Load velocity field (from PIV or other analysis)
velocity_data = np.load('velocity_field.npz')
x = velocity_data['x']  # Grid coordinates (μm), shape [ny, nx]
y = velocity_data['y']
u = velocity_data['u']  # Velocity (μm/s), shape [nt, ny, nx]
v = velocity_data['v']
```

### Step 2: Create Preprocessor

The preprocessor handles background subtraction and normalization:

```python
from fastvideo.preprocessing_unified import from_stack_analysis

# Automatic preprocessing from stack
preprocessor = from_stack_analysis(
    stack,
    n_frames=30,      # Average first 30 frames for background
    sigma_low=100,    # Gaussian smoothing width
    vmin_percentile=0,
    vmax_percentile=100
)

# Test on a single frame
processed = preprocessor(stack[100].data)
```

### Step 3: Preview Before Rendering

Always preview to verify settings:

```python
from fastvideo.preview import preview_single_frame

# Quick preview
preview = preview_single_frame(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,    # Microns per pixel
    frame_idx=100,
    save_path='preview.png'
)

# Display
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(preview)
plt.title('Video Preview')
plt.axis('off')
plt.show()
```

### Step 4: Create Video

```python
from fastvideo.video_creator import create_velocity_video

create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,           # μm/pixel from microscope
    output_path='bacteria_flow.mp4',
    target_size_mb=200,        # Desired file size
    fps=25,                    # Playback speed
    use_gpu=True,              # 10-20× faster
    verbose=True
)
```

## Complete Example

Here's a complete script from start to finish:

```python
#!/usr/bin/env python3
"""
Create microscopy video with velocity overlay
"""

import numpy as np
import matplotlib.pyplot as plt
from cam import loadstack

from fastvideo.preprocessing_unified import from_stack_analysis
from fastvideo.preview import preview_single_frame
from fastvideo.video_creator import create_velocity_video

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("Loading data...")
stack = loadstack('bacteria_experiment.zip')
velocity_data = np.load('velocity_field.npz')

x = velocity_data['x']  # Coordinates (μm)
y = velocity_data['y']
u = velocity_data['u']  # Velocity (μm/s)
v = velocity_data['v']

print(f"Stack: {len(stack)} frames")
print(f"Velocity grid: {u.shape}")

# ============================================================================
# 2. CREATE PREPROCESSOR
# ============================================================================

print("\nCreating preprocessor...")
preprocessor = from_stack_analysis(
    stack,
    n_frames=30,
    sigma_low=100,
    vmin_percentile=1,   # Clip 1% darkest pixels
    vmax_percentile=99   # Clip 1% brightest pixels
)

# ============================================================================
# 3. PREVIEW
# ============================================================================

print("\nGenerating preview...")
preview = preview_single_frame(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    frame_idx=100,
    output_resolution=(1024, 1024),  # Downsample for smaller file
    save_path='video_preview.png'
)

plt.figure(figsize=(10, 10))
plt.imshow(preview)
plt.title('Video Preview - Frame 100')
plt.axis('off')
plt.tight_layout()
plt.savefig('preview_display.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. CREATE VIDEO
# ============================================================================

print("\nCreating video...")
create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    output_path='bacteria_flow.mp4',
    target_size_mb=200,
    fps=25,
    codec='h265',        # Best compression
    start=0,
    end=None,            # All frames
    skip=1,              # Every frame
    use_gpu=True,
    verbose=True
)

print("\n✓ Video created successfully!")
print("Output: bacteria_flow.mp4")
```

## Common Adjustments

### Adjust Video Duration

```python
# Make video 2× faster (skip every other frame)
create_velocity_video(
    ...,
    skip=2,
    fps=25  # Playback fps stays the same
)

# Process only frames 100-500
create_velocity_video(
    ...,
    start=100,
    end=500,
    skip=1
)
```

### Adjust Arrow Appearance

```python
from fastvideo.preview import interactive_preview

# Interactively tune parameters
optimal_params = interactive_preview(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65
)

# Use optimal settings
create_velocity_video(
    ...,
    arrow_scale=optimal_params['arrow_scale'],
    subsample=optimal_params['subsample']
)
```

### Change Video Quality

```python
# Higher quality (larger file)
create_velocity_video(
    ...,
    target_size_mb=500,
    codec='h265',
    output_resolution=None  # Full resolution
)

# Lower quality (smaller file)
create_velocity_video(
    ...,
    target_size_mb=100,
    codec='h264',  # Less efficient compression
    output_resolution=(512, 512)
)
```

## GPU Acceleration

### Enable GPU Processing

```python
# Check if GPU is available
try:
    import cupy as cp
    print(f"GPU available: {cp.cuda.runtime.getDeviceCount()} device(s)")
except ImportError:
    print("CuPy not installed - using CPU only")

# Use GPU
create_velocity_video(
    ...,
    use_gpu=True  # 10-20× faster
)
```

### GPU Memory Management

For very large datasets:

```python
import cupy as cp

# Process in chunks
n_chunks = 4
frames_per_chunk = len(stack) // n_chunks

for i in range(n_chunks):
    start = i * frames_per_chunk
    end = (i + 1) * frames_per_chunk
    
    create_velocity_video(
        stack=stack,
        preprocessor=preprocessor,
        x=x, y=y, u=u, v=v,
        pixel_size=0.65,
        output_path=f'chunk_{i+1}.mp4',
        start=start,
        end=end,
        use_gpu=True
    )
    
    # Clear GPU memory between chunks
    cp.get_default_memory_pool().free_all_blocks()

# Merge chunks with FFmpeg
# ffmpeg -i "concat:chunk_0.mp4|chunk_1.mp4|chunk_2.mp4|chunk_3.mp4" -c copy output.mp4
```

## Validation Before Rendering

Save time by validating settings first:

```python
from fastvideo.preview import validate_video_settings

report = validate_video_settings(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    target_size_mb=200,
    fps=25,
    start=0,
    end=1000,
    skip=1
)

# Check results
if report['is_valid']:
    print("✓ Settings are valid!")
    # Proceed with video creation
else:
    print("✗ Fix these errors:")
    for error in report['errors']:
        print(f"  - {error}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
```

## Troubleshooting

### Problem: Video is all black

```python
# Check preprocessing output
img = preprocessor(stack[0].data)
print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")

# Should be in range [0, 1] or [0, 255]
# If not, adjust:
preprocessor.vmin = 0.0
preprocessor.vmax = 1.0
preprocessor.normalize = True
```

### Problem: Arrows not visible

```python
# Use interactive preview to adjust
from fastvideo.preview import interactive_preview

params = interactive_preview(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65
)

# Use returned parameters
create_velocity_video(
    ...,
    arrow_scale=params['arrow_scale'],
    subsample=params['subsample']
)
```

### Problem: File too large

```python
from fastvideo.utils import calculate_optimal_resolution

# Calculate optimal resolution
width, height, bitrate = calculate_optimal_resolution(
    original_width=1648,
    original_height=1648,
    target_size_mb=200,
    duration_seconds=40,
    fps=25,
    codec='h265'
)

print(f"Use resolution: {width}×{height}")

# Apply to video creation
create_velocity_video(
    ...,
    output_resolution=(width, height)
)
```

### Problem: FFmpeg errors

```python
# Check codec availability
from fastvideo.ffmpeg_writer import check_ffmpeg_codecs

codecs = check_ffmpeg_codecs()
print("Available codecs:")
for codec, available in codecs.items():
    print(f"  {codec}: {'✓' if available else '✗'}")

# If h265 not available, use h264
create_velocity_video(
    ...,
    codec='h264'  # More compatible
)
```

## Next Steps

- Read the [full README](README.md) for advanced features
- Explore custom preprocessing in `preprocessing_unified.py`
- Check nematic field analysis in `nematic.py`
- Use plotting utilities in `fast_plot.py`

## Getting Help

If you encounter issues:

1. Run validation: `validate_video_settings(...)`
2. Generate preview: `preview_single_frame(...)`
3. Check preprocessing: `img = preprocessor(stack[0].data); print(img.min(), img.max())`
4. Verify FFmpeg: `ffmpeg -version`
5. Open an issue on GitHub with error messages

---

**Happy video creation!** 🎥
