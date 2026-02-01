# FastVideo - Scientific Microscopy Video Creation

A comprehensive, production-ready Python package for creating high-quality videos from microscopy image stacks with optional velocity field overlays.

## Features

- **Multiple codec support**: H264, H265, XVID, MJPEG with automatic fallback
- **GPU acceleration**: CUDA-accelerated processing when CuPy is available  
- **Quality control**: Automatic parameter optimization and validation
- **Vector field overlays**: Fast PIV (Particle Image Velocimetry) visualization
- **Robust error handling**: Graceful fallbacks and comprehensive diagnostics
- **Clean API**: Simple functions for common tasks, full control for advanced use

## Installation

```bash
# Basic installation (CPU only)
pip install opencv-python numpy tqdm

# With GPU support (optional)
pip install cupy-cuda11x  # or appropriate CUDA version

# For full functionality
pip install matplotlib scipy
```

## Quick Start

### Basic Video Creation

```python
from fastvideo import video_from_stack

# Define preprocessing function
def preprocess(img):
    """Normalize image to [0, 1]"""
    return (img - img.min()) / (img.max() - img.min())

# Create video
video_from_stack(
    stack,                    # Your image stack
    preprocess,               # Preprocessing function
    'output.mp4',            # Output filename
    speed=2.0,               # 2x playback speed
    codec='h264',            # H264 codec
    quality='high',          # High quality
    text=True                # Show time overlay
)
```

### Velocity Field Video

```python
from fastvideo import create_velocity_video

# Assuming you have PIV results: x, y, u, v
create_velocity_video(
    stack, preprocess, x, y, u, v,
    filename='velocity.mp4',
    pixel_size=0.65,         # Microns per pixel
    vector_scale=None,       # Auto-calibrate arrows
    vector_skip=2,           # Draw every 2nd arrow
    vector_color=(255, 255, 0)  # Yellow arrows
)
```

### Auto-Calibrated Arrows

```python
from fastvideo import calculate_auto_arrow_scale, video_with_vectors

# Calculate optimal arrow scale
arrow_scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)
print(f"Recommended scale: {arrow_scale:.2f}")

# Use in video creation
video_with_vectors(
    stack, preprocess, x, y, u, v, 'piv.mp4',
    vector_scale=arrow_scale,
    vector_skip=3
)
```

## API Reference

### Core Functions

#### `video_from_stack()`
Create grayscale video from microscopy stack.

**Parameters:**
- `stack`: Stack object with `.data` and `.time()` methods
- `preprocess`: Function to preprocess frames
- `filename`: Output video filename
- `speed`: Playback speed multiplier (default: 1.0)
- `codec`: Video codec - 'h264', 'h265', 'mp4v', etc. (default: 'h264')
- `quality`: Quality preset - 'high', 'medium', 'low' (default: 'medium')
- `skip`: Frame skip factor (default: 1)
- `text`: Add time overlay (default: False)

#### `video_with_vectors()`
Create video with velocity field overlay.

**Parameters:**
- `stack`, `preprocess`, `filename`: Same as above
- `x, y`: Coordinate meshgrids (2D arrays)
- `u, v`: Velocity components (2D or 3D with time)
- `vector_skip`: Draw every N-th vector (default: 2)
- `vector_scale`: Arrow length scaling (default: 1.0)
- `vector_color`: RGB color tuple (default: yellow)
- `vector_thickness`: Arrow thickness (default: 2)

#### `create_velocity_video()`
High-level function with auto-calibration.

**Parameters:**
- Same as `video_with_vectors()` plus:
- `pixel_size`: Physical size per pixel (required for auto-scaling)

### Utility Functions

#### `calculate_auto_arrow_scale(u, v, pixel_size)`
Calculate optimal arrow scale for visualization.

**Returns:** float - Recommended scale factor

#### `check_available_codecs()`
Test which video codecs are available on your system.

**Returns:** list - Available codec names

#### `normalize_image(img, output_range='uint8')`
Normalize image to specified range.

**Parameters:**
- `img`: Input image (any dtype)
- `output_range`: 'uint8' or 'float'

**Returns:** Normalized image

## Advanced Usage

### Custom Preprocessing

```python
from scipy.ndimage import gaussian_filter

def advanced_preprocess(img):
    """Advanced preprocessing with background subtraction."""
    # Smooth
    smooth = gaussian_filter(img, sigma=1.5)
    
    # Background subtraction
    background = gaussian_filter(smooth, sigma=50)
    subtracted = smooth - background
    
    # Normalize
    result = (subtracted - subtracted.min())
    result = result / result.max()
    
    return result

video_from_stack(stack, advanced_preprocess, 'processed.mp4')
```

### RGB Color Videos

```python
import matplotlib.pyplot as plt

def rgb_preprocess(img):
    """Convert to RGB using colormap."""
    img_norm = (img - img.min()) / (img.max() - img.min())
    rgb = plt.cm.viridis(img_norm)[:, :, :3]  # Drop alpha channel
    return (rgb * 255).astype(np.uint8)

from fastvideo import video_from_stack_color
video_from_stack_color(stack, rgb_preprocess, 'color.mp4')
```

### Frame Subset

```python
# Process only frames 100-500, using every 2nd frame
video_from_stack(
    stack, preprocess, 'subset.mp4',
    start=100,
    end=500,
    skip=2
)
```

### Manual FPS Control

```python
# Override automatic FPS calculation
video_from_stack(
    stack, preprocess, 'custom_fps.mp4',
    fps=60.0  # Force 60 FPS
)
```

## Examples

### Example 1: Basic Microscopy Video

```python
import numpy as np
from fastvideo import video_from_stack

# Load your stack (example structure)
# stack[i].data contains image at index i
# stack.time() returns array of timestamps

def simple_normalize(img):
    return img / img.max()

video_from_stack(
    stack, 
    simple_normalize, 
    'bacteria_movie.mp4',
    speed=5.0,           # 5x speed
    quality='high',
    text=True,           # Show timestamps
    fontsize=1.5
)
```

### Example 2: PIV Visualization

```python
from fastvideo import create_velocity_video

# After running PIV analysis to get u, v fields
create_velocity_video(
    stack, 
    lambda x: x/x.max(), 
    x_coords, y_coords, u_field, v_field,
    filename='piv_flow.mp4',
    pixel_size=0.65,              # 0.65 microns/pixel
    vector_color=(0, 255, 0),     # Green arrows
    vector_skip=3,                # Every 3rd arrow
    quality='high',
    text=True
)
```

### Example 3: Checking System Capabilities

```python
from fastvideo import check_available_codecs, get_system_info

# Check what codecs are available
available = check_available_codecs()

# Get full system information
info = get_system_info()
print(info)
```

## Troubleshooting

### No codecs available
```bash
# Install FFmpeg with codec support
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from ffmpeg.org
```

### GPU not detected
```bash
# Install CuPy with correct CUDA version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

### Video plays too fast/slow
```python
# Adjust speed parameter
video_from_stack(stack, preprocess, 'output.mp4', speed=0.5)  # Slower
video_from_stack(stack, preprocess, 'output.mp4', speed=2.0)  # Faster
```

### Arrows too long/short
```python
# Use auto-calibration
from fastvideo import calculate_auto_arrow_scale
scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)

# Or manually adjust
video_with_vectors(..., vector_scale=2.5)  # Longer arrows
video_with_vectors(..., vector_scale=0.5)  # Shorter arrows
```

## Performance Tips

1. **Use frame skipping** for large datasets:
   ```python
   video_from_stack(stack, preprocess, 'output.mp4', skip=2)
   ```

2. **Reduce vector density** for faster rendering:
   ```python
   video_with_vectors(..., vector_skip=4)  # Draw every 4th vector
   ```

3. **Use appropriate quality** for your needs:
   ```python
   # Preview: quality='low'
   # Final: quality='high'
   ```

4. **Consider H265** for better compression:
   ```python
   video_from_stack(..., codec='h265')  # Smaller file size
   ```

## Requirements

- Python >= 3.7
- NumPy >= 1.18
- OpenCV-Python >= 4.0
- tqdm >= 4.50

Optional:
- CuPy >= 8.0 (for GPU acceleration)
- Matplotlib >= 3.0 (for preview features in full package)
- SciPy >= 1.5 (for advanced preprocessing)

## License

MIT License - see LICENSE file for details.

## Author

Carlo - Scientific microscopy video processing

## Version

2.0.0

---

For more examples and advanced features, see the `examples/` directory and full documentation.
