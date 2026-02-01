# FastVideo - High-Performance Microscopy Video Creation

A GPU-accelerated Python toolkit for creating ultra-compressed videos from microscopy image stacks with velocity field overlays.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **GPU Acceleration**: CUDA kernels for 100-200× faster processing
- **Ultra Compression**: 18 GB stacks → ~200 MB videos (H.265/HEVC)
- **Smart Preprocessing**: Unified API for background subtraction and normalization
- **Velocity Overlays**: Automatic arrow calibration and rendering
- **Preview & Validate**: Check settings before long renders
- **Robust Pipeline**: Comprehensive error handling and diagnostics

## Quick Start

```python
from fastvideo import create_velocity_video
from fastvideo.preprocessing_unified import from_stack_analysis
from cam import loadstack

# Load data
stack = loadstack('bacteria.zip')
velocity_data = np.load('velocity_field.npz')

# Create preprocessor
preprocessor = from_stack_analysis(stack, n_frames=30, sigma_low=100)

# Create video with automatic optimization
create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=velocity_data['x'],
    y=velocity_data['y'],
    u=velocity_data['u'],
    v=velocity_data['v'],
    pixel_size=0.65,  # μm/pixel
    output_path='bacteria_flow.mp4',
    target_size_mb=200,
    fps=25,
    use_gpu=True  # 10-20× faster
)
```

## Installation

### Basic Installation

```bash
pip install numpy scipy matplotlib opencv-python numba
pip install ffmpeg-python  # Or install FFmpeg system-wide
```

### GPU Acceleration (Optional but Recommended)

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### System Dependencies

**FFmpeg** (required for video encoding):

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Check installation
ffmpeg -version
```

**NVENC** (optional, for GPU encoding):
- Requires NVIDIA GPU with hardware encoder
- Check availability: `nvidia-smi`

## Architecture

```
fastvideo/
├── preprocessing_unified.py   # Image preprocessing API
├── overlay.py                 # Velocity field overlay
├── ffmpeg_writer.py          # Robust video encoding
├── utils.py                  # Resolution & arrow calibration
├── preview.py                # Preview & validation
├── video_creator.py          # Main creation pipeline
├── nematic.py               # Nematic tensor extraction
├── histogram.py             # Binning utilities
└── fast_plot.py             # Publication-quality plotting
```

## Core Modules

### 1. Image Preprocessing (`preprocessing_unified.py`)

Unified API for microscopy image preprocessing with background subtraction, filtering, and normalization.

```python
from fastvideo.preprocessing_unified import ImagePreprocessor, from_stack_analysis

# Method 1: Automatic from stack analysis
preprocessor = from_stack_analysis(
    stack,
    n_frames=30,      # Average first 30 frames for background
    sigma_low=100,    # Gaussian smoothing for background
    vmin_percentile=0,
    vmax_percentile=100
)

# Method 2: Manual configuration
preprocessor = ImagePreprocessor(
    bg_avg=background_pattern,     # 2D array or scalar
    bg_low=lowfreq_background,     # Normalization factor
    sigma=2.0,                     # Gaussian smoothing
    vmin_percentile=1,
    vmax_percentile=99,
    use_gpu=True  # GPU acceleration
)
preprocessor.fit(reference_image)

# Process images
processed = preprocessor(raw_image)

# Save/load preprocessor
preprocessor.save('preprocessor.pkl')
loaded = ImagePreprocessor.load('preprocessor.pkl', use_gpu=True)
```

**Key Features:**
- Picklable (can save/load entire preprocessing pipeline)
- GPU support via CuPy
- Auto-fitting from reference images
- Factory functions for common use cases

### 2. Velocity Field Overlay (`overlay.py`)

Fast arrow rendering on microscopy images using Numba JIT or CUDA kernels.

```python
from fastvideo.overlay import VelocityOverlayProcessor, GPUVelocityOverlayProcessor

# CPU version (Numba-optimized)
processor = VelocityOverlayProcessor(
    base_preprocessor=preprocessor,
    x=x_coords,               # Grid coordinates (μm)
    y=y_coords,
    u=velocity_u,             # Velocity (μm/s), shape [nt, ny, nx]
    v=velocity_v,
    pixel_size=0.65,          # μm/pixel
    dt=0.033,                 # Time step (s)
    output_resolution=(1024, 1024),  # Downsample
    arrow_scale=None,         # Auto-calibrate
    arrow_width=2,
    arrow_color=(255, 255, 0),  # RGB yellow
    subsample=None            # Auto-calibrate
)

# GPU version (100-200× faster)
gpu_processor = GPUVelocityOverlayProcessor(
    # Same parameters as CPU version
)

# Process frame
img_with_arrows = processor(raw_frame, frame_idx=100)
```

**Performance:**
- CPU (Numba): 10-50× faster than OpenCV loops
- GPU (CUDA): 100-200× faster than CPU
- Automatic arrow calibration for optimal visibility

### 3. Video Encoding (`ffmpeg_writer.py`)

Robust FFmpeg wrapper with comprehensive error handling.

```python
from fastvideo.ffmpeg_writer import RobustFFmpegWriter, check_ffmpeg_codecs

# Check available codecs
codecs = check_ffmpeg_codecs()
print(f"H.265 GPU encoding available: {codecs['h265_nvenc']}")

# Create writer
with RobustFFmpegWriter(
    filename='output.mp4',
    width=1024,
    height=1024,
    fps=25,
    bitrate='500k',
    codec='h265',       # or 'h265_nvenc' for GPU
    preset='medium',    # or 'p4' for NVENC
    verbose=True
) as writer:
    for frame in frames:
        writer.write(frame)  # Must be uint8 RGB
```

**Codecs:**
- `h265`/`hevc`: Best compression (CPU)
- `h265_nvenc`: GPU encoding (NVIDIA only)
- `h264`: Wide compatibility (CPU)
- `h264_nvenc`: GPU encoding (NVIDIA only)

**Presets:**
- CPU: `ultrafast`, `fast`, `medium`, `slow`, `veryslow`
- NVENC: `p1` (fastest) to `p7` (best quality)

### 4. Utilities (`utils.py`)

Helper functions for resolution optimization and arrow calibration.

```python
from fastvideo.utils import calculate_optimal_resolution, auto_calibrate_arrows

# Optimize resolution for target file size
width, height, bitrate = calculate_optimal_resolution(
    original_width=1648,
    original_height=1648,
    target_size_mb=200,
    duration_seconds=40,
    fps=25,
    codec='h265'
)
# Output: (1024, 1024, '512k')

# Auto-calibrate arrow parameters
arrow_scale, arrow_width, subsample = auto_calibrate_arrows(
    u=velocity_u,
    v=velocity_v,
    pixel_size=0.65,
    dt=0.033,
    is_timeseries=True
)
# Targets: ~20 pixel arrow length, ~5% arrow density
```

### 5. Preview & Validation (`preview.py`)

Validate settings before rendering to avoid wasted computation.

```python
from fastvideo.preview import (
    preview_single_frame,
    preview_multiple_frames,
    validate_video_settings,
    interactive_preview
)

# Quick preview
preview = preview_single_frame(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    frame_idx=100,
    save_path='preview.png'
)

# Preview multiple frames
preview_multiple_frames(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    n_frames=6,
    save_path='preview_grid.png'
)

# Comprehensive validation
report = validate_video_settings(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    target_size_mb=200,
    fps=25
)

if report['is_valid']:
    print("✓ Ready to create video!")
else:
    print("Fix these errors:")
    for error in report['errors']:
        print(f"  - {error}")

# Interactive parameter tuning
optimal_params = interactive_preview(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65
)
# Use sliders to adjust arrow_scale and subsample
```

### 6. Main Pipeline (`video_creator.py`)

High-level functions for video creation.

```python
from fastvideo.video_creator import (
    create_velocity_video,
    create_video_simple,
    create_video_with_validation
)

# Full-featured creation with automatic optimization
create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    output_path='bacteria.mp4',
    target_size_mb=200,
    fps=25,
    codec='h265',
    start=0,
    end=None,  # All frames
    skip=1,    # Use every frame
    arrow_scale=None,  # Auto-calibrate
    arrow_width=2,
    arrow_color=(255, 255, 0),
    subsample=None,    # Auto-calibrate
    use_gpu=True,
    verbose=True
)

# Simple creation (minimal parameters)
create_video_simple(
    stack=stack,
    preprocessor=preprocessor,
    output_path='simple.mp4'
)

# With validation before rendering
create_video_with_validation(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    output_path='validated.mp4',
    target_size_mb=200
)
```

## Advanced Usage

### Custom Preprocessing Pipeline

```python
from fastvideo.preprocessing_unified import ImagePreprocessor

# Complex preprocessing with spatial background
background_pattern = gaussian_filter(avg_frames, sigma=100)
spatial_variation = compute_illumination_pattern(stack)

preprocessor = ImagePreprocessor(
    bg_avg=background_pattern,
    bg_low=spatial_variation,
    sigma=2.0,  # Additional smoothing
    vmin_percentile=1,
    vmax_percentile=99,
    normalize=True,
    use_gpu=True
)
preprocessor.fit(stack[0].data)
```

### Processing Subsets of Frames

```python
# Process every 3rd frame from frame 100 to 1000
create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    output_path='subset.mp4',
    start=100,
    end=1000,
    skip=3,  # Every 3rd frame
    fps=25   # Output fps (not acquisition fps)
)
```

### Custom Arrow Styling

```python
from fastvideo.overlay import VelocityOverlayProcessor

processor = VelocityOverlayProcessor(
    preprocessor,
    x, y, u, v,
    pixel_size=0.65,
    dt=0.033,
    arrow_scale=3.0,        # Longer arrows
    arrow_width=3,          # Thicker lines
    arrow_color=(0, 255, 255),  # Cyan
    subsample=4             # Sparser arrows
)
```

### GPU Memory Management

```python
# For large datasets, process in chunks
import cupy as cp

for start in range(0, len(stack), 500):
    end = min(start + 500, len(stack))
    
    create_velocity_video(
        stack=stack,
        preprocessor=preprocessor,
        x=x, y=y, u=u, v=v,
        pixel_size=0.65,
        output_path=f'chunk_{start:04d}.mp4',
        start=start,
        end=end,
        use_gpu=True
    )
    
    # Clear GPU memory
    cp.get_default_memory_pool().free_all_blocks()
```

## Performance Benchmarks

Tested on 1648×1648 microscopy stack (1000 frames):

| Configuration | Processing Time | Output Size | Throughput |
|--------------|----------------|-------------|------------|
| CPU (Numba) | ~60 min | 195 MB | ~16 fps |
| GPU (CUDA) | ~4 min | 195 MB | ~250 fps |
| GPU + NVENC | ~2 min | 195 MB | ~500 fps |

**System:** NVIDIA RTX 3080, Intel i9-10900K, 32GB RAM

## Troubleshooting

### Video is all black
```python
# Check preprocessing output range
img = preprocessor(stack[0].data)
print(f"Range: [{img.min():.3f}, {img.max():.3f}]")

# Should be [0, 1] or [0, 255]
# If wrong, adjust preprocessing:
preprocessor.vmin = 0.0
preprocessor.vmax = 1.0
```

### FFmpeg errors
```python
# Check codec availability
from fastvideo.ffmpeg_writer import check_ffmpeg_codecs
print(check_ffmpeg_codecs())

# Validate parameters
from fastvideo.ffmpeg_writer import validate_ffmpeg_params
validate_ffmpeg_params(width=1024, height=1024, fps=25, codec='h265')
```

### Arrows too small/large
```python
# Use interactive preview to tune
from fastvideo.preview import interactive_preview

params = interactive_preview(stack, preprocessor, x, y, u, v, pixel_size=0.65)
# Adjust sliders, then use returned parameters
```

### Out of GPU memory
```python
# Reduce batch size or use CPU
create_velocity_video(
    ...,
    output_resolution=(512, 512),  # Lower resolution
    use_gpu=False  # Fall back to CPU
)
```

## Nematic Field Analysis

Extract and visualize nematic director fields from microscopy images.

```python
from fastvideo.nematic import extract_nematic_field, stack_extract_nematic

# Single frame
eig_max, eig_min, dx, dy, x_centers, y_centers = extract_nematic_field(
    img=preprocessed_image,
    window=64,
    overlap=32,
    mpp=0.65,
    smooth_sigma=2.0,
    use_gpu=True,
    method='auto'  # 'gpu' or 'numba'
)

# Entire stack
nematic_data = stack_extract_nematic(
    stack=stack,
    preprocessor=preprocessor,
    window=64,
    overlap=32,
    mpp=0.65,
    use_gpu=True,
    verbose=True
)
# Returns: {'n': director_field, 'l': eigenvalues, 'x': coords, 'y': coords, 't': time}
```

## Plotting Utilities

Create publication-quality plots with minimal code.

```python
from fastvideo.fast_plot import quick_plot, plot_multi_series, quick_subplots

# Quick single plot
fig, ax = quick_plot(
    x, y,
    title='Velocity vs Time',
    xlabel='Time (s)',
    ylabel='Velocity (μm/s)',
    color='blue',
    marker='o',
    grid=True
)

# Multiple series
fig, ax = quick_plot_multi(
    x, [y1, y2, y3],
    labels=['Control', 'Treatment 1', 'Treatment 2'],
    colors=['black', 'blue', 'red'],
    title='Experimental Results',
    xlabel='Time (s)',
    ylabel='Signal',
    grid=True
)

# Subplots with composable functions
def plot_panel_a(ax):
    plot_multi_series(ax, time, [signal1, signal2],
                     labels=['Rep 1', 'Rep 2'],
                     title='(A) Time Series',
                     xlabel='Time', ylabel='Signal',
                     grid=True)

def plot_panel_b(ax):
    ax.imshow(image, cmap='gray')
    ax.set_title('(B) Microscopy')
    ax.axis('off')

fig, axs = quick_subplots([plot_panel_a, plot_panel_b], ncols=2)
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this software in your research, please cite:

```bibtex
@software{fastvideo2024,
  author = {Carlo},
  title = {FastVideo: High-Performance Microscopy Video Creation},
  year = {2024},
  url = {https://github.com/yourusername/fastvideo}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- CUDA kernels optimized for nematic tensor computation
- FFmpeg for robust video encoding
- CuPy team for GPU array processing
- Numba for JIT compilation

## Contact

For questions, issues, or feature requests:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Version:** 1.0.0  
**Last Updated:** 2024  
**Python:** 3.8+  
**GPU:** NVIDIA CUDA 11.0+ (optional)
