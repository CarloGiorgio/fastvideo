# Video Creation Package for Microscopy Analysis

Comprehensive video generation from microscopy image stacks with support for velocity field overlays, multiple codecs, and automatic parameter optimization.

## Features

- **High-quality video encoding** with multiple codec support (H.264, H.265, MP4V, etc.)
- **Grayscale and RGB video creation** from image stacks
- **Vector field overlays** for PIV and velocity visualization
- **GPU acceleration** support (when CuPy is available)
- **Automatic parameter calibration** for arrow scaling and resolution
- **FFmpeg integration** for advanced encoding options
- **Preview and validation** before full rendering

## Installation

The package requires:
- Python 3.7+
- NumPy
- OpenCV (cv2)
- tqdm

Optional dependencies:
- CuPy (for GPU acceleration)
- matplotlib (for preview functions)
- ffmpeg (for advanced encoding)

## Quick Start

### Basic Video Creation

```python
from _video import video_from_stack

# Define preprocessing function
def preprocess(img):
    return img / img.max()  # Normalize to [0, 1]

# Create video
video_from_stack(
    stack, 
    preprocess, 
    'output.mp4',
    speed=2.0,      # 2x playback speed
    codec='h264',   # H.264 codec
    quality='high'  # High quality
)
```

### Velocity Field Video

```python
from _video import video_with_vectors

# Create PIV visualization video
video_with_vectors(
    stack, preprocess, x, y, u, v,
    'velocity.mp4',
    vector_skip=3,          # Draw every 3rd arrow
    vector_scale=2.0,       # Arrow length scale
    vector_color=(255, 255, 0),  # Yellow arrows
    text=True,              # Show time overlay
    fps=30
)
```

### Automatic Parameter Calibration

```python
from _video import auto_calibrate_arrows, create_velocity_video

# Auto-calibrate arrow parameters
params = auto_calibrate_arrows(u, v, pixel_size=0.65)

# Create video with calibrated parameters
create_velocity_video(
    stack, preprocess, x, y, u, v,
    'calibrated.mp4',
    pixel_size=0.65,
    vector_scale=params['scale'],
    vector_skip=params['subsample']
)
```

## Main Functions

### Basic Video Creation

- **`video_from_stack`**: Create grayscale video from stack
- **`video_from_stack_color`**: Create RGB video from stack
- **`create_video_simple`**: High-level wrapper with defaults

### Vector Field Videos

- **`video_with_vectors`**: Create video with velocity overlay
- **`create_velocity_video`**: Auto-calibrated velocity video
- **`draw_vectors_opencv`**: Draw vector field on image

### Utilities

- **`calculate_fps`**: Calculate frame rate from timing
- **`auto_calibrate_arrows`**: Calibrate arrow parameters
- **`calculate_optimal_resolution`**: Optimize resolution for file size
- **`check_available_codecs`**: Test available video codecs

### Preview & Validation

- **`preview_single_frame`**: Preview one frame
- **`preview_multiple_frames`**: Preview multiple frames
- **`validate_video_settings`**: Validate parameters

## API Reference

### video_from_stack

Create grayscale video from microscopy stack.

**Parameters:**
- `stack`: Stack object with `.data` and `.time()` methods
- `preprocess`: Function to preprocess each frame
- `filename`: Output video filename
- `speed`: Playback speed multiplier (default: 1.0)
- `skip`: Frame skip factor (default: 1)
- `codec`: Video codec - 'h264', 'h265', 'mp4v' (default: 'h264')
- `quality`: Quality preset - 'high', 'medium', 'low' (default: 'medium')
- `fps`: Manual FPS override (optional)
- `start`, `end`: Frame range (default: all frames)
- `text`: Add time overlay (default: False)

### video_with_vectors

Create video with vector field overlay.

**Parameters:**
- `stack`, `preprocess`, `filename`: Same as above
- `x`, `y`: Meshgrid coordinates (2D arrays)
- `u`, `v`: Velocity components (shape: n_frames × n_rows × n_cols)
- `vector_skip`: Draw every N-th vector (default: 2)
- `vector_scale`: Arrow length scaling (default: 1.0)
- `vector_color`: RGB color tuple (default: yellow)
- `vector_thickness`: Arrow line width (default: 2)
- Additional parameters same as `video_from_stack`

### auto_calibrate_arrows

Automatically calibrate arrow visualization parameters.

**Parameters:**
- `u`, `v`: Velocity components
- `pixel_size`: Physical size per pixel
- `target_length_pixels`: Target arrow length (default: 20)

**Returns:**
- Dictionary with 'scale', 'subsample', 'thickness'

## Advanced Usage

### Custom Preprocessing

```python
from scipy.ndimage import gaussian_filter

def custom_preprocess(img):
    # Apply Gaussian filter
    img_smooth = gaussian_filter(img, sigma=2.0)
    
    # Normalize
    img_norm = (img_smooth - img_smooth.min()) / (img_smooth.max() - img_smooth.min())
    
    return img_norm

video_from_stack(stack, custom_preprocess, 'smooth.mp4')
```

### Color Video Creation

```python
def color_preprocess(img, frame_idx):
    # Create RGB from grayscale
    norm = img / img.max()
    
    # Map to color (e.g., hot colormap)
    r = norm
    g = norm * 0.5
    b = norm * 0.2
    
    return np.stack([r, g, b], axis=-1)

video_from_stack_color(stack, color_preprocess, 'color.mp4')
```

### Resolution Optimization

```python
from _video import calculate_optimal_resolution, estimate_file_size

# Check estimated file size
original_size = (1920, 1080)
duration = 60  # seconds
fps = 30

size_mb = estimate_file_size(*original_size, duration, fps, codec='h264')
print(f"Estimated size: {size_mb:.1f} MB")

# Optimize for target size
optimal_size = calculate_optimal_resolution(
    original_size, 
    target_size_mb=100, 
    duration_s=duration,
    fps=fps
)
print(f"Optimized resolution: {optimal_size}")
```

## Codec Selection

### Available Codecs

Test which codecs are available:

```python
from _video import check_available_codecs

check_available_codecs()
```

### Codec Comparison

| Codec | Compression | Speed | Quality | Use Case |
|-------|-------------|-------|---------|----------|
| h264 | Medium | Fast | Good | General use, best compatibility |
| h265 | High | Slower | Excellent | High compression, smaller files |
| mp4v | Low | Very fast | Fair | Quick previews |
| xvid | Medium | Medium | Good | Alternative to H.264 |
| mjpg | Very low | Fastest | Poor | Debug/testing only |

**Recommendation**: Use 'h264' for most cases, 'h265' for maximum compression.

## Performance Tips

1. **Use frame skipping** for faster rendering: `skip=2` doubles speed
2. **Reduce vector density**: `vector_skip=4` for large fields
3. **Lower resolution** for previews: Use smaller spatial dimensions
4. **GPU acceleration**: Install CuPy for faster processing
5. **Codec choice**: Use 'mjpg' for fast test renders

## Troubleshooting

### "Codec not available"
- Install ffmpeg with codec support: `conda install ffmpeg`
- Try alternative codec: `codec='mp4v'`

### "Video file not created"
- Check write permissions in output directory
- Verify filename has correct extension (.mp4, .avi)
- Ensure codec is available (run `check_available_codecs()`)

### Slow rendering
- Increase `skip` parameter
- Reduce `vector_skip` for arrow videos
- Use GPU acceleration if available
- Lower video quality: `quality='low'`

### Large file sizes
- Use H.265 codec: `codec='h265'`
- Reduce quality: `quality='medium'` or `'low'`
- Scale down resolution
- Increase frame skip

## Examples

See `_video/examples.py` for complete working examples.

## License

Part of the SLMicro microscopy analysis package.

## Author

Carlo - Bacterial Active Matter Research
