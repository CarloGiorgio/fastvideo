# FastVideo API Reference

Complete API documentation for all modules in the fastvideo package.

## Table of Contents

1. [preprocessing_unified](#preprocessing_unified)
2. [overlay](#overlay)
3. [ffmpeg_writer](#ffmpeg_writer)
4. [utils](#utils)
5. [preview](#preview)
6. [video_creator](#video_creator)
7. [nematic](#nematic)
8. [histogram](#histogram)
9. [fast_plot](#fast_plot)

---

## preprocessing_unified

Unified preprocessing API for microscopy image analysis.

### Classes

#### `ImagePreprocessor`

Main preprocessing class with background subtraction, filtering, and normalization.

```python
ImagePreprocessor(
    bg_avg=None,          # Average background (float or ndarray)
    bg_low=None,          # Low-frequency background (float or ndarray)
    sigma=None,           # Gaussian filter sigma
    vmin_percentile=0,    # Lower clipping percentile (0-100)
    vmax_percentile=100,  # Upper clipping percentile (0-100)
    normalize=True,       # Normalize to [0, 1]
    use_gpu=False         # Use GPU acceleration
)
```

**Methods:**

- `fit(reference_image)` - Compute clipping range from reference
- `set_clip_range(vmin, vmax)` - Manually set clipping range
- `__call__(img)` - Process an image
- `save(filename)` - Save to pickle file
- `load(filename, use_gpu=False)` - Load from pickle file

**Example:**
```python
prep = ImagePreprocessor(bg_avg=100.0, sigma=2.0)
prep.fit(reference_image)
processed = prep(raw_image)
prep.save('preprocessor.pkl')
```

### Factory Functions

#### `from_stack_analysis()`

Create preprocessor by analyzing initial frames of a stack.

```python
from_stack_analysis(
    stack,                 # Stack object with .data attribute
    n_frames=30,          # Frames to average for background
    sigma_low=100,        # Gaussian sigma for background smoothing
    vmin_percentile=0,    # Lower percentile for clipping
    vmax_percentile=100,  # Upper percentile for clipping
    normalize=True,       # Normalize output
    use_gpu=False         # GPU acceleration
)
```

**Returns:** Fitted `ImagePreprocessor`

**Example:**
```python
prep = from_stack_analysis(stack, n_frames=30, sigma_low=100)
processed = prep(stack[100].data)
```

#### `from_pattern_subtraction()`

Create preprocessor for pattern/background subtraction.

```python
from_pattern_subtraction(
    pattern,              # Background pattern (ndarray)
    sigma=None,           # Optional Gaussian smoothing
    vmin_percentile=0,
    vmax_percentile=100,
    normalize=True,
    use_gpu=False
)
```

**Example:**
```python
pattern = np.load('background.npy')
prep = from_pattern_subtraction(pattern, sigma=2.0)
prep.fit(reference_image)
```

#### `from_evidence_detection()`

Create preprocessor for feature detection (high-pass filtering).

```python
from_evidence_detection(
    reference_image,      # Reference for clipping range
    sigma,                # Gaussian sigma (features < sigma are highlighted)
    vmin_percentile=0,
    vmax_percentile=100,
    use_gpu=False
)
```

**Returns:** Fitted `ImagePreprocessor` for feature detection

**Example:**
```python
# Highlight features smaller than 50 pixels
prep = from_evidence_detection(stack[0].data, sigma=50, vmin_percentile=5)
```

---

## overlay

Velocity field overlay processors with CPU (Numba) and GPU (CUDA) implementations.

### Classes

#### `VelocityOverlayProcessor`

CPU-based velocity overlay using Numba JIT compilation.

```python
VelocityOverlayProcessor(
    base_preprocessor,     # Callable: preprocess(img) -> normalized_img
    x,                     # Grid x-coordinates (μm), shape [ny, nx]
    y,                     # Grid y-coordinates (μm), shape [ny, nx]
    u,                     # Velocity u-component (μm/s), [nt, ny, nx] or [ny, nx]
    v,                     # Velocity v-component (μm/s), [nt, ny, nx] or [ny, nx]
    pixel_size,            # Microns per pixel (float)
    dt,                    # Time step (seconds)
    output_resolution=None,  # (width, height) or None
    arrow_scale=None,      # Arrow length multiplier (None = auto)
    arrow_width=2,         # Arrow line width (pixels)
    arrow_color=(255, 255, 0),  # RGB tuple (0-255)
    subsample=None,        # Arrow subsampling (None = auto)
    interpolation='area'   # 'area' or 'linear' for downsampling
)
```

**Methods:**

- `__call__(img, frame_idx=0)` - Process image with velocity overlay

**Returns:** RGB image with arrows, uint8 [0, 255], shape (H, W, 3)

**Example:**
```python
processor = VelocityOverlayProcessor(
    preprocessor, x, y, u, v,
    pixel_size=0.65,
    dt=0.033,
    output_resolution=(1024, 1024),
    arrow_scale=2.0,
    subsample=2
)

img_with_arrows = processor(raw_frame, frame_idx=100)
```

#### `GPUVelocityOverlayProcessor`

GPU-accelerated version using CUDA kernels (100-200× faster).

```python
GPUVelocityOverlayProcessor(
    # Same parameters as VelocityOverlayProcessor
)
```

**Requirements:** CuPy, NVIDIA GPU with CUDA support

**Example:**
```python
gpu_processor = GPUVelocityOverlayProcessor(
    preprocessor, x, y, u, v,
    pixel_size=0.65,
    dt=0.033,
    use_gpu=True
)

img_with_arrows = gpu_processor(raw_frame, frame_idx=100)
```

---

## ffmpeg_writer

Robust FFmpeg video encoding with comprehensive error handling.

### Functions

#### `check_ffmpeg_codecs()`

Check which video codecs are available.

```python
check_ffmpeg_codecs()
```

**Returns:** Dictionary with codec availability

**Example:**
```python
codecs = check_ffmpeg_codecs()
# {'h264': True, 'h265': True, 'h264_nvenc': False, 'h265_nvenc': False}
```

#### `validate_ffmpeg_params()`

Validate FFmpeg parameters before starting encoding.

```python
validate_ffmpeg_params(
    width,    # int
    height,   # int
    fps,      # float
    codec     # str
)
```

**Returns:** True if valid, False otherwise

### Classes

#### `RobustFFmpegWriter`

FFmpeg video writer with error handling.

```python
RobustFFmpegWriter(
    filename,          # Output file path
    width,             # Frame width (pixels)
    height,            # Frame height (pixels)
    fps,               # Frames per second
    bitrate='500k',    # Target bitrate ('500k', '2M', etc.)
    codec='h265',      # 'h264', 'h265', 'h264_nvenc', 'h265_nvenc'
    preset='medium',   # CPU: 'ultrafast' to 'veryslow', NVENC: 'p1' to 'p7'
    pix_fmt='yuv420p', # Pixel format
    verbose=False      # Print FFmpeg output
)
```

**Methods:**

- `write(frame)` - Write a frame (must be uint8 RGB)
- `close()` - Finalize video
- `__enter__()`, `__exit__()` - Context manager support

**Example:**
```python
with RobustFFmpegWriter('output.mp4', 1024, 1024, 25, codec='h265') as writer:
    for frame in frames:
        writer.write(frame)
```

---

## utils

Utility functions for resolution optimization and arrow calibration.

### Functions

#### `calculate_optimal_resolution()`

Calculate optimal resolution to achieve target file size.

```python
calculate_optimal_resolution(
    original_width,      # int
    original_height,     # int
    target_size_mb,      # float
    duration_seconds,    # float
    fps,                 # float
    codec='h265'         # str
)
```

**Returns:** `(width, height, bitrate)`

**Example:**
```python
width, height, bitrate = calculate_optimal_resolution(
    1648, 1648,
    target_size_mb=200,
    duration_seconds=40,
    fps=25
)
# (1024, 1024, '512k')
```

#### `auto_calibrate_arrows()`

Automatically calibrate arrow visualization parameters.

```python
auto_calibrate_arrows(
    u,                    # Velocity u-component (μm/s)
    v,                    # Velocity v-component (μm/s)
    pixel_size,           # Microns per pixel
    dt,                   # Time step (seconds)
    is_timeseries=None    # Auto-detected if None
)
```

**Returns:** `(arrow_scale, arrow_width, subsample)`

**Example:**
```python
arrow_scale, arrow_width, subsample = auto_calibrate_arrows(
    u, v,
    pixel_size=0.65,
    dt=0.033
)
# Targets: ~20 pixel arrows, ~5% density
```

#### `mm2inch()`

Convert millimeters to inches for figure sizing.

```python
mm2inch(*tupl)  # float or tuple of floats
```

**Example:**
```python
figsize = mm2inch(100, 80)  # (3.937, 3.150)
```

#### `estimate_file_size()`

Estimate output video file size.

```python
estimate_file_size(
    width,       # int
    height,      # int
    n_frames,    # int
    fps,         # float
    codec='h265' # str
)
```

**Returns:** Estimated size in megabytes

---

## preview

Preview and validation functions.

### Functions

#### `preview_single_frame()`

Preview a single frame with velocity overlay.

```python
preview_single_frame(
    stack,                # Stack object
    preprocessor,         # Callable
    x, y,                 # Grid coordinates (μm)
    u, v,                 # Velocity (μm/s)
    pixel_size,           # float
    frame_idx=0,          # int
    output_resolution=None,  # (width, height) or None
    arrow_scale=None,     # float or None (auto)
    arrow_width=2,        # int
    arrow_color=(255, 255, 0),  # RGB tuple
    subsample=None,       # int or None (auto)
    save_path=None,       # str or None
    use_gpu=False         # bool
)
```

**Returns:** Preview image (H, W, 3) uint8

**Example:**
```python
preview = preview_single_frame(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    frame_idx=100,
    save_path='preview.png'
)
```

#### `preview_multiple_frames()`

Preview multiple frames in a grid.

```python
preview_multiple_frames(
    stack, preprocessor, x, y, u, v, pixel_size,
    frame_indices=None,   # List of indices or None
    n_frames=4,           # int (if frame_indices is None)
    output_resolution=None,
    arrow_scale=None,
    arrow_width=2,
    arrow_color=(255, 255, 0),
    subsample=None,
    figsize=(16, 8),
    save_path=None,
    use_gpu=False
)
```

**Example:**
```python
preview_multiple_frames(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    frame_indices=[0, 100, 500, 999],
    save_path='preview_grid.png'
)
```

#### `validate_video_settings()`

Comprehensive validation of video settings.

```python
validate_video_settings(
    stack, preprocessor, x, y, u, v, pixel_size,
    output_resolution=None,
    arrow_scale=None,
    arrow_width=2,
    arrow_color=(255, 255, 0),
    subsample=None,
    target_size_mb=200,
    fps=25,
    codec='h265',
    start=0,
    end=None,
    skip=1,
    use_gpu=False
)
```

**Returns:** Dictionary with validation report

**Keys:**
- `is_valid` - bool
- `warnings` - list of warning messages
- `errors` - list of error messages
- `info` - dict with diagnostic information
- `recommendations` - list of suggested improvements

**Example:**
```python
report = validate_video_settings(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    target_size_mb=200
)

if report['is_valid']:
    print("✓ Ready to render!")
else:
    for error in report['errors']:
        print(f"✗ {error}")
```

#### `interactive_preview()`

Interactive preview with adjustable sliders.

```python
interactive_preview(
    stack, preprocessor, x, y, u, v, pixel_size,
    output_resolution=None,
    use_gpu=False
)
```

**Returns:** Dictionary with optimal parameters

**Example:**
```python
params = interactive_preview(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65
)
# Use sliders to adjust, then:
# params = {'arrow_scale': 2.5, 'subsample': 2}
```

---

## video_creator

High-level video creation pipeline.

### Functions

#### `create_velocity_video()`

Full-featured video creation with automatic optimization.

```python
create_velocity_video(
    stack,                # Stack object
    preprocessor,         # Callable
    x, y,                 # Grid coordinates (μm), shape [ny, nx]
    u, v,                 # Velocity (μm/s), shape [nt, ny, nx] or [ny, nx]
    pixel_size,           # Microns per pixel
    output_path,          # str
    target_size_mb=200,   # float
    fps=25,               # float
    codec='h265',         # str
    start=0,              # int
    end=None,             # int or None (all frames)
    skip=1,               # int (use every Nth frame)
    arrow_scale=None,     # float or None (auto)
    arrow_width=2,        # int
    arrow_color=(255, 255, 0),  # RGB tuple
    subsample=None,       # int or None (auto)
    resolution_scale=None,  # float or None (auto)
    use_gpu=False,        # bool
    verbose=True          # bool
)
```

**Example:**
```python
create_velocity_video(
    stack=stack,
    preprocessor=preprocessor,
    x=x, y=y, u=u, v=v,
    pixel_size=0.65,
    output_path='bacteria.mp4',
    target_size_mb=200,
    fps=25,
    use_gpu=True
)
```

#### `create_video_simple()`

Minimal parameter video creation.

```python
create_video_simple(
    stack,
    preprocessor,
    output_path,
    fps=25,
    use_gpu=False
)
```

#### `create_video_with_validation()`

Create video with automatic validation before rendering.

```python
create_video_with_validation(
    stack, preprocessor, x, y, u, v, pixel_size,
    output_path,
    target_size_mb=200,
    fps=25,
    # ... other parameters
)
```

**Example:**
```python
create_video_with_validation(
    stack, preprocessor, x, y, u, v,
    pixel_size=0.65,
    output_path='validated.mp4',
    target_size_mb=200
)
# Automatically runs validation first
```

---

## nematic

Nematic tensor extraction from microscopy images.

### Functions

#### `extract_nematic_field()`

Extract nematic director field from a single image.

```python
extract_nematic_field(
    img,                  # Preprocessed image (H, W)
    window=64,            # Window size (pixels)
    overlap=32,           # Stride (pixels)
    mpp=1.0,              # Microns per pixel
    smooth_sigma=None,    # Gaussian smoothing before gradients
    traceless=True,       # Use traceless tensor formulation
    use_gpu=False,        # GPU acceleration
    method='auto'         # 'auto', 'gpu', or 'numba'
)
```

**Returns:** `(eig_max, eig_min, dx, dy, x_centers, y_centers)`

**Example:**
```python
eig_max, eig_min, dx, dy, x, y = extract_nematic_field(
    preprocessed_image,
    window=64,
    overlap=32,
    mpp=0.65,
    use_gpu=True
)
```

#### `stack_extract_nematic()`

Extract nematic fields from multiple frames.

```python
stack_extract_nematic(
    stack,                # Stack object
    preprocessor,         # Callable
    window=64,
    overlap=32,
    mpp=1.0,
    smooth_sigma=None,
    traceless=True,
    use_gpu=False,
    method='auto',
    frame_indices=None,   # Indices to process (None = all)
    verbose=True
)
```

**Returns:** Dictionary with keys:
- `'n'`: director field, shape (n_frames, 2, n_y, n_x)
- `'l'`: eigenvalues, shape (n_frames, 2, n_y, n_x)
- `'x'`: x coordinates, shape (n_y, n_x)
- `'y'`: y coordinates, shape (n_y, n_x)
- `'t'`: time array, shape (n_frames,)

**Example:**
```python
nematic_data = stack_extract_nematic(
    stack, preprocessor,
    window=64,
    overlap=32,
    mpp=0.65,
    use_gpu=True
)

# Save results
np.savez('nematic_field.npz', **nematic_data)
```

---

## histogram

Binning and downsampling utilities.

### Functions

#### `binning()`

Bin data in linear or log scale.

```python
binning(
    template,    # Template values (what to bin by)
    x,           # Data x-values
    y,           # Data y-values
    nob=50,      # Number of bins
    log=False    # Logarithmic binning
)
```

**Returns:** `(hx, hy)` - binned values

#### `block_average()`

Perform block averaging on n-dimensional array.

```python
block_average(
    data,            # nd.array
    block_size=(1,1,1)  # tuple
)
```

**Example:**
```python
# Average 3D data in 2×2×2 blocks
averaged = block_average(data_3d, block_size=(2, 2, 2))
```

---

## fast_plot

Publication-quality plotting utilities.

### Quick Plot Functions

#### `quick_plot()`

Create standalone line plot.

```python
quick_plot(
    x, y,
    figsize=(10, 6),
    # Plot parameters
    color=None, marker=None, linestyle='-', linewidth=2,
    # Formatting
    title='', xlabel='', ylabel='',
    xlim=None, ylim=None,
    grid=False, legend=False,
    logx=False, logy=False
)
```

**Returns:** `(fig, ax)`

#### `quick_plot_multi()`

Plot multiple data series.

```python
quick_plot_multi(
    x,              # Shared x or list of x arrays
    y_list,         # List of y arrays
    figsize=(10, 6),
    labels=None,    # List of labels
    colors=None,    # List of colors
    markers=None,   # List of markers or single marker
    linestyles='-', # List or single linestyle
    # ... formatting parameters
)
```

**Example:**
```python
fig, ax = quick_plot_multi(
    time, [signal1, signal2, signal3],
    labels=['Control', 'Treatment 1', 'Treatment 2'],
    colors=['black', 'blue', 'red'],
    title='Experimental Results',
    grid=True
)
```

### Composable Plot Functions

#### `plot_multi_series()`

Plot multiple series on existing axes (composable).

```python
plot_multi_series(
    ax,             # Matplotlib axes
    x,              # Shared x-axis
    y_list,         # List of y arrays
    labels=None,
    colors=None,
    markers='o',
    linestyles='-',
    linewidth=2,
    markersize=6,
    alpha=1.0,
    show_legend=True,
    legend_loc='best',
    # Formatting kwargs
    title='', xlabel='', ylabel='',
    grid=False, logx=False, logy=False
)
```

**Example:**
```python
def my_plot(ax):
    plot_multi_series(
        ax, time, [y1, y2],
        labels=['A', 'B'],
        title='My Data',
        grid=True
    )

fig, axs = quick_subplots([my_plot], ncols=1)
```

#### `quick_subplots()`

Create subplots and apply plotting functions.

```python
quick_subplots(
    plot_funcs,          # List of callables: func(ax)
    nrows=1,
    ncols=2,
    figsize=(14, 6),
    width_ratios=None,
    height_ratios=None,
    suptitle='',
    tight_layout=True
)
```

**Returns:** `(fig, axs)`

**Example:**
```python
def plot1(ax):
    ax.plot(x1, y1)
    ax.set_title('Panel A')

def plot2(ax):
    ax.scatter(x2, y2)
    ax.set_title('Panel B')

fig, axs = quick_subplots([plot1, plot2], ncols=2)
```

### Utility Functions

#### `mm2inch()`

Convert millimeters to inches.

```python
mm2inch(*tupl)  # float or tuple
```

#### `extract_colors_cmap()`

Extract colors from colormap.

```python
extract_colors_cmap(
    colormap='viridis',
    vmin=0.1,
    vmax=1.0,
    n=5
)
```

**Returns:** `(cmap, colors)` - colormap and array of RGBA colors

---

## Common Patterns

### Processing Pipeline

```python
# 1. Load data
stack = loadstack('data.zip')
velocity = np.load('velocity.npz')

# 2. Create preprocessor
preprocessor = from_stack_analysis(stack)

# 3. Validate
report = validate_video_settings(
    stack, preprocessor,
    velocity['x'], velocity['y'], velocity['u'], velocity['v'],
    pixel_size=0.65
)

# 4. Create video
if report['is_valid']:
    create_velocity_video(
        stack, preprocessor,
        velocity['x'], velocity['y'], velocity['u'], velocity['v'],
        pixel_size=0.65,
        output_path='output.mp4'
    )
```

### GPU Acceleration

```python
# Check GPU availability
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False

# Use GPU when available
create_velocity_video(
    ...,
    use_gpu=gpu_available
)
```

### Custom Arrow Styling

```python
# Interactive tuning
params = interactive_preview(stack, preprocessor, x, y, u, v, pixel_size=0.65)

# Apply custom style
create_velocity_video(
    ...,
    arrow_scale=params['arrow_scale'],
    arrow_width=3,
    arrow_color=(0, 255, 255),  # Cyan
    subsample=params['subsample']
)
```

---

For more examples, see [QUICKSTART.md](QUICKSTART.md) and [README.md](README.md).
