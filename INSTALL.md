# FastVideo Package - Installation & Usage Guide

## Package Contents

Your `fastvideo` package contains all the video creation functionality consolidated into a clean, production-ready structure:

```
fastvideo/
├── Core Modules (Ready to Use)
│   ├── __init__.py           # Main package interface
│   ├── video_writer.py       # VideoWriter class with codec management
│   ├── video_utils.py        # Utility functions
│   ├── video_basic.py        # Basic video creation (grayscale & RGB)
│   ├── video_vectors.py      # Vector field visualization
│   └── video_creator.py      # High-level convenience functions
│
├── Advanced Modules (Stubs - use full _video for these)
│   ├── preview.py            # Preview & validation
│   ├── overlay.py            # Overlay processors
│   ├── batch.py              # Batch processing
│   ├── ffmpeg_integration.py # FFmpeg integration
│   └── utils.py              # Additional utilities
│
├── Documentation
│   ├── README.md             # Complete user guide
│   ├── API.md                # Full API reference
│   ├── QUICKSTART.md         # 5-minute tutorial
│   ├── PACKAGE_OVERVIEW.md   # Architecture guide
│   └── examples.py           # 12 working examples
│
└── Installation
    ├── setup.py              # Installation script
    ├── LICENSE               # MIT license
    └── CHANGELOG.md          # Version history
```

## Installation

### Method 1: Direct Installation (Recommended)

```bash
cd fastvideo/
pip install -e .
```

This installs the package in "editable" mode so you can modify it.

### Method 2: Regular Installation

```bash
cd fastvideo/
pip install .
```

### Method 3: Add to Python Path

If you don't want to install, just add to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/fastvideo')
import fastvideo
```

### Dependencies

The installer will automatically install:
- `numpy >= 1.18`
- `opencv-python >= 4.0`
- `tqdm >= 4.50`

Optional (install separately if needed):
```bash
# For GPU acceleration
pip install cupy-cuda11x  # or cupy-cuda12x

# For full features
pip install matplotlib scipy
```

## Quick Start

### 1. Basic Video

```python
from fastvideo import video_from_stack

# Define preprocessing
def preprocess(img):
    return (img - img.min()) / (img.max() - img.min())

# Create video
video_from_stack(
    stack,              # Your image stack
    preprocess,         # Preprocessing function
    'output.mp4',       # Output filename
    speed=2.0,          # 2x playback
    quality='high'      # High quality
)
```

### 2. Velocity Field Video

```python
from fastvideo import create_velocity_video

# Assuming you have PIV results: x, y, u, v
create_velocity_video(
    stack, preprocess, x, y, u, v,
    filename='piv_video.mp4',
    pixel_size=0.65,         # Microns per pixel
    vector_scale=None        # Auto-calibrate arrows!
)
```

### 3. Custom Arrow Styling

```python
from fastvideo import video_with_vectors, calculate_auto_arrow_scale

# Calculate optimal scale
scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)

# Create video with custom style
video_with_vectors(
    stack, preprocess, x, y, u, v, 'custom.mp4',
    vector_scale=scale,
    vector_skip=3,              # Draw every 3rd arrow
    vector_color=(0, 255, 0),   # Green arrows
    vector_thickness=3          # Thick arrows
)
```

## Verification

After installation, verify it works:

```python
import fastvideo

# Check version
print(f"FastVideo version: {fastvideo.__version__}")

# Check system
fastvideo.check_available_codecs()

# Get system info
info = fastvideo.get_system_info()
print(info)
```

## Available Functions

### Core Functions

```python
from fastvideo import (
    # Basic video creation
    video_from_stack,          # Grayscale video
    video_from_stack_color,    # RGB video
    
    # Vector field videos
    video_with_vectors,        # Full control
    create_velocity_video,     # Auto-calibrated
    
    # Quick creation
    create_video_simple,       # Simple with defaults
    
    # Utilities
    calculate_auto_arrow_scale,  # Auto arrow calibration
    check_available_codecs,      # Check system codecs
    normalize_image,             # Image normalization
    add_text_overlay,           # Add text to images
    draw_vectors_opencv,        # Draw arrows on image
    
    # Low-level
    VideoWriter,               # Video writer class
)
```

### Example Usage

```python
# List all available functions
import fastvideo
fastvideo.list_functions()

# Show example code
fastvideo.show_examples()

# Run examples (if you have a stack)
from fastvideo.examples import example_basic_video
example_basic_video(stack)
```

## Common Workflows

### Workflow 1: Quick Video

```python
from fastvideo import create_video_simple

create_video_simple(
    stack,
    lambda x: x / x.max(),
    'quick_video.mp4'
)
```

### Workflow 2: High-Quality with Text

```python
from fastvideo import video_from_stack

video_from_stack(
    stack,
    lambda x: x / x.max(),
    'annotated.mp4',
    speed=1.0,
    quality='high',
    text=True,
    fontsize=1.5,
    text_color=(255, 255, 0)
)
```

### Workflow 3: PIV Analysis Video

```python
from fastvideo import create_velocity_video

# After PIV analysis
create_velocity_video(
    stack, preprocess, x, y, u, v,
    'piv_analysis.mp4',
    pixel_size=0.65,
    quality='high',
    text=True
)
```

### Workflow 4: Batch Processing

```python
# Process multiple videos with different speeds
speeds = [0.5, 1.0, 2.0, 5.0]

for speed in speeds:
    video_from_stack(
        stack, preprocess,
        f'video_speed_{speed}x.mp4',
        speed=speed,
        quality='medium'
    )
```

## Troubleshooting

### Issue 1: "No codec available"

**Solution:** Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows: Download from ffmpeg.org
```

### Issue 2: Import Error

```python
# If you get: ModuleNotFoundError: No module named 'fastvideo'

# Make sure you installed it:
pip install -e /path/to/fastvideo/

# Or add to path:
import sys
sys.path.insert(0, '/path/to/fastvideo')
```

### Issue 3: Arrows Too Long/Short

```python
# Auto-calibrate
from fastvideo import calculate_auto_arrow_scale
scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)

# Use the calculated scale
video_with_vectors(..., vector_scale=scale)
```

### Issue 4: GPU Not Detected

```bash
# Install CuPy with correct CUDA version
pip install cupy-cuda11x  # For CUDA 11
pip install cupy-cuda12x  # For CUDA 12
```

## Documentation

All documentation is included:

1. **README.md** - Start here for overview and examples
2. **QUICKSTART.md** - 5-minute getting started guide
3. **API.md** - Complete API reference for all functions
4. **PACKAGE_OVERVIEW.md** - Architecture and design
5. **examples.py** - 12 complete working examples

Read them in order: QUICKSTART → README → API → examples.py

## Advanced Usage

For advanced features (preview, batch processing, FFmpeg integration), you'll need the complete `_video` package. The current `fastvideo` package includes stubs for these that will guide you.

## Support

If you need help:

1. Check the documentation in this order:
   - QUICKSTART.md (quick answers)
   - README.md (comprehensive guide)
   - API.md (detailed function reference)
   - examples.py (code examples)

2. Run diagnostics:
   ```python
   import fastvideo
   print(fastvideo.get_system_info())
   fastvideo.check_available_codecs()
   ```

3. Look at the examples:
   ```python
   from fastvideo.examples import example_check_system
   example_check_system()
   ```

## Next Steps

1. **Install the package:**
   ```bash
   cd fastvideo/
   pip install -e .
   ```

2. **Test it:**
   ```python
   import fastvideo
   fastvideo.check_available_codecs()
   ```

3. **Read the quickstart:**
   ```bash
   cat fastvideo/QUICKSTART.md
   ```

4. **Try an example:**
   ```python
   from fastvideo import video_from_stack
   
   def preprocess(img):
       return img / img.max()
   
   video_from_stack(stack, preprocess, 'test.mp4')
   ```

5. **Explore the API:**
   ```bash
   cat fastvideo/API.md
   ```

---

## Summary

The `fastvideo` package is ready to use! It provides:

✅ Clean, well-documented API  
✅ Automatic codec fallback  
✅ Auto-calibrated velocity videos  
✅ GPU support (optional)  
✅ Comprehensive examples  
✅ Production-ready structure  

Install it with `pip install -e fastvideo/` and start creating videos!

---

**Package:** fastvideo v2.0.0  
**Author:** Carlo  
**License:** MIT  
**Status:** Production Ready ✨
