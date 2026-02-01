# FastVideo Quickstart Guide

Get started with FastVideo in 5 minutes!

## Installation

```bash
# Basic installation
pip install opencv-python numpy tqdm

# With GPU support (optional)
pip install cupy-cuda11x  # Adjust for your CUDA version
```

## 30-Second Example

```python
from fastvideo import video_from_stack

# Define preprocessing
def preprocess(img):
    return img / img.max()

# Create video
video_from_stack(stack, preprocess, 'output.mp4')
```

Done! You now have a video at `output.mp4`.

## Common Use Cases

### 1. Basic Microscopy Video

```python
from fastvideo import video_from_stack

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

video_from_stack(
    stack,
    normalize,
    'bacteria.mp4',
    speed=2.0,        # 2x playback
    quality='high'
)
```

### 2. PIV (Velocity Field) Video

```python
from fastvideo import create_velocity_video

# You have: stack, x, y, u, v from PIV analysis

create_velocity_video(
    stack, normalize, x, y, u, v,
    'piv_video.mp4',
    pixel_size=0.65    # microns/pixel
)
```

The arrows are automatically calibrated!

### 3. Custom Arrows

```python
from fastvideo import video_with_vectors

video_with_vectors(
    stack, normalize, x, y, u, v, 'custom.mp4',
    vector_skip=3,              # Draw every 3rd arrow
    vector_scale=2.0,           # Manual scale
    vector_color=(0, 255, 0),   # Green arrows
    vector_thickness=3          # Thicker arrows
)
```

### 4. Time Overlay

```python
video_from_stack(
    stack, normalize, 'with_time.mp4',
    text=True,                  # Show time
    fontsize=1.5,
    text_color=(255, 255, 0),   # Yellow text
    text_position=(50, 50)
)
```

### 5. Process Subset

```python
# Only frames 100-500, every 2nd frame
video_from_stack(
    stack, normalize, 'subset.mp4',
    start=100,
    end=500,
    skip=2
)
```

## Troubleshooting

### "No codec available"

```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

### Arrows too long/short?

```python
from fastvideo import calculate_auto_arrow_scale

# Get recommended scale
scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)
print(f"Use scale={scale:.2f}")

# Use it
video_with_vectors(..., vector_scale=scale)
```

### Video plays too fast?

```python
# Slow down
video_from_stack(..., speed=0.5)  # Half speed
```

### Need better quality?

```python
video_from_stack(..., quality='high')  # High quality
video_from_stack(..., codec='h265')    # Better compression
```

## Next Steps

- Read the [README](README.md) for detailed documentation
- Check [examples.py](examples.py) for more examples
- See [API.md](API.md) for complete API reference

## Quick Reference

```python
# Check system
from fastvideo import check_available_codecs
check_available_codecs()

# Simple video
from fastvideo import create_video_simple
create_video_simple(stack, preprocess, 'out.mp4')

# Velocity video
from fastvideo import create_velocity_video
create_velocity_video(stack, preprocess, x, y, u, v, 'piv.mp4', pixel_size=0.65)

# Full control
from fastvideo import video_with_vectors
video_with_vectors(stack, preprocess, x, y, u, v, 'custom.mp4', **kwargs)
```

---

Happy video creation! 🎬
