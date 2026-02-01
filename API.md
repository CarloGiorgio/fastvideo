# FastVideo API Reference

Complete API documentation for all public functions and classes.

## Table of Contents

1. [Core Video Creation](#core-video-creation)
2. [Vector Field Videos](#vector-field-videos)
3. [High-Level Functions](#high-level-functions)
4. [Utilities](#utilities)
5. [Video Writer](#video-writer)

---

## Core Video Creation

### `video_from_stack()`

Create grayscale video from microscopy image stack.

**Signature:**
```python
video_from_stack(
    stack,
    preprocess: Callable,
    filename: str,
    speed: float = 1.0,
    skip: int = 1,
    codec: str = 'h264',
    quality: str = 'medium',
    dt: Optional[float] = None,
    fps: Optional[float] = None,
    start: int = 0,
    end: int = -1,
    text: bool = False,
    fontsize: float = 2.0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_position: Tuple[int, int] = (50, 50),
    verbose: bool = True
) -> None
```

**Parameters:**

- **stack** (object): Stack object with `.data` attribute and `.time()` method
- **preprocess** (callable): Function `preprocess(img) -> normalized_img` that returns 2D array
- **filename** (str): Output video filename (e.g., 'output.mp4')
- **speed** (float): Playback speed multiplier. 2.0 = 2x speed
- **skip** (int): Frame skip factor. 2 = use every 2nd frame
- **codec** (str): Video codec. Options: 'h264', 'h265', 'mp4v', 'xvid', 'mjpg'
- **quality** (str): Quality preset. Options: 'high', 'medium', 'low'
- **dt** (float, optional): Time spacing override. Auto-calculated if None
- **fps** (float, optional): Manual FPS override. Ignores speed/dt if set
- **start** (int): Starting frame index
- **end** (int): Ending frame index. -1 = use all frames
- **text** (bool): Add time text overlay
- **fontsize** (float): Text font scale
- **text_color** (tuple): Text RGB color
- **text_position** (tuple): Text position (x, y) in pixels
- **verbose** (bool): Print progress information

**Returns:** None (writes video to disk)

**Examples:**
```python
from fastvideo import video_from_stack

def preprocess(img):
    return (img - img.min()) / (img.max() - img.min())

video_from_stack(stack, preprocess, 'output.mp4', speed=2.0, quality='high')
```

---

### `video_from_stack_color()`

Create RGB color video from microscopy stack.

Similar to `video_from_stack()` but preprocessing function must return RGB image with shape `(height, width, 3)`.

**Signature:**
```python
video_from_stack_color(
    stack,
    preprocess: Callable,  # Must return RGB!
    filename: str,
    # ... same parameters as video_from_stack
) -> None
```

**Examples:**
```python
import matplotlib.pyplot as plt

def rgb_preprocess(img):
    img_norm = (img - img.min()) / (img.max() - img.min())
    rgba = plt.cm.viridis(img_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)

video_from_stack_color(stack, rgb_preprocess, 'color.mp4')
```

---

## Vector Field Videos

### `video_with_vectors()`

Create video with velocity field overlay.

**Signature:**
```python
video_with_vectors(
    stack,
    preprocess: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    filename: str,
    speed: float = 1.0,
    skip: int = 1,
    codec: str = 'h264',
    quality: str = 'medium',
    dt: Optional[float] = None,
    fps: Optional[float] = None,
    start: int = 0,
    end: int = -1,
    vector_skip: int = 2,
    vector_scale: float = 1.0,
    vector_color: Tuple[int, int, int] = (255, 255, 0),
    vector_thickness: int = 2,
    text: bool = False,
    fontsize: float = 2.0,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    text_position: Tuple[int, int] = (50, 50),
    use_gpu: bool = False,
    verbose: bool = True
) -> None
```

**Parameters:**

All parameters from `video_from_stack()` plus:

- **x, y** (np.ndarray): Coordinate meshgrids (2D arrays)
- **u, v** (np.ndarray): Velocity components. Shape: `(n_frames, rows, cols)` for time-series or `(rows, cols)` for static
- **vector_skip** (int): Draw every N-th vector
- **vector_scale** (float): Arrow length scaling factor
- **vector_color** (tuple): Arrow RGB color. Default: yellow (255, 255, 0)
- **vector_thickness** (int): Arrow line thickness in pixels
- **use_gpu** (bool): Use GPU acceleration if available

**Examples:**
```python
video_with_vectors(
    stack, preprocess, x, y, u, v, 'piv.mp4',
    vector_skip=3,
    vector_scale=2.0,
    vector_color=(0, 255, 0)  # Green arrows
)
```

---

### `draw_vectors_opencv()`

Draw velocity vectors on a single image.

**Signature:**
```python
draw_vectors_opencv(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    skip: int = 2,
    scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 2,
    arrow_tip: float = 0.3
) -> np.ndarray
```

**Parameters:**

- **img** (np.ndarray): Background image (grayscale or RGB)
- **x, y** (np.ndarray): Coordinate meshgrids
- **u, v** (np.ndarray): Velocity components (2D, same shape as x, y)
- **skip** (int): Draw every N-th vector
- **scale** (float): Vector length scaling
- **color** (tuple): Arrow RGB color
- **thickness** (int): Arrow thickness
- **arrow_tip** (float): Arrow tip size ratio (0-1)

**Returns:** RGB image with arrows (uint8)

**Examples:**
```python
from fastvideo import draw_vectors_opencv

img_with_arrows = draw_vectors_opencv(
    img, x, y, u, v,
    skip=4, scale=2.5, color=(255, 0, 0)
)
```

---

## High-Level Functions

### `create_video_simple()`

Quick video creation with sensible defaults.

**Signature:**
```python
create_video_simple(
    stack,
    preprocess: Callable,
    filename: str,
    **kwargs
) -> None
```

Automatically sets: `codec='h264'`, `quality='medium'`, `speed=1.0`

**Examples:**
```python
from fastvideo import create_video_simple

create_video_simple(stack, lambda x: x/x.max(), 'quick.mp4')
```

---

### `create_velocity_video()`

Create velocity video with automatic arrow calibration.

**Signature:**
```python
create_velocity_video(
    stack,
    preprocess: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    filename: str,
    pixel_size: float,
    **kwargs
) -> None
```

**Key Features:**
- Automatically calculates `vector_scale` from `pixel_size` if not provided
- Sets sensible defaults for arrow display
- Prints recommended scale value

**Examples:**
```python
from fastvideo import create_velocity_video

create_velocity_video(
    stack, preprocess, x, y, u, v,
    'auto_piv.mp4',
    pixel_size=0.65  # microns/pixel
)
```

---

## Utilities

### `calculate_auto_arrow_scale()`

Calculate optimal arrow scale for visualization.

**Signature:**
```python
calculate_auto_arrow_scale(
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    target_length_pixels: float = 20.0
) -> float
```

**Parameters:**

- **u, v** (np.ndarray): Velocity components
- **pixel_size** (float): Physical size per pixel
- **target_length_pixels** (float): Target arrow length for median velocity

**Returns:** Recommended scale factor (float)

**Examples:**
```python
from fastvideo import calculate_auto_arrow_scale

scale = calculate_auto_arrow_scale(u, v, pixel_size=0.65)
print(f"Use scale={scale:.2f}")
```

---

### `calculate_fps()`

Calculate frames per second from timing information.

**Signature:**
```python
calculate_fps(
    times: np.ndarray,
    speed: float = 1.0,
    skip: int = 1,
    dt: Optional[float] = None
) -> float
```

**Returns:** FPS value (minimum 1.0)

---

### `normalize_image()`

Normalize image to specified output range.

**Signature:**
```python
normalize_image(
    img: np.ndarray,
    output_range: str = 'uint8'
) -> np.ndarray
```

**Parameters:**

- **img** (np.ndarray): Input image (any dtype, any range)
- **output_range** (str): 'uint8' (0-255) or 'float' (0-1)

**Returns:** Normalized image

---

### `add_text_overlay()`

Add text overlay to image.

**Signature:**
```python
add_text_overlay(
    img: np.ndarray,
    text: str,
    position: Tuple[int, int] = (100, 100),
    fontsize: float = 2.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    background: bool = True,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    background_alpha: float = 0.7
) -> np.ndarray
```

**Returns:** Image with text overlay

---

### `check_available_codecs()`

Test which video codecs are available.

**Signature:**
```python
check_available_codecs(verbose: bool = True) -> List[str]
```

**Returns:** List of available codec names

**Examples:**
```python
from fastvideo import check_available_codecs

codecs = check_available_codecs()
print(f"Available: {codecs}")
```

---

## Video Writer

### `VideoWriter`

Low-level video writer with automatic codec management.

**Signature:**
```python
class VideoWriter:
    def __init__(
        self,
        filename: str,
        width: int,
        height: int,
        fps: float,
        is_color: bool = True,
        codec: str = 'h264',
        quality: str = 'medium'
    )
```

**Methods:**

- `write(frame: np.ndarray)`: Write a frame
- `release()`: Close the writer
- `get_stats() -> dict`: Get video statistics

**Context Manager:**
```python
with VideoWriter('out.mp4', 1920, 1080, 30.0) as writer:
    for frame in frames:
        writer.write(frame)
```

**Examples:**
```python
from fastvideo import VideoWriter

writer = VideoWriter('output.mp4', 640, 480, 30.0, is_color=False)
for frame in frames:
    writer.write(frame)
writer.release()
```

---

## Type Hints

Common type definitions used throughout the API:

```python
from typing import Callable, Optional, Tuple
import numpy as np

# Preprocessing function type
PreprocessFunc = Callable[[np.ndarray], np.ndarray]

# RGB color type
RGBColor = Tuple[int, int, int]

# Position type
Position = Tuple[int, int]
```

---

## Error Handling

All functions may raise:

- **ValueError**: Invalid parameters (e.g., negative dimensions, invalid frame range)
- **RuntimeError**: Video writer initialization failed, codec not available
- **FileNotFoundError**: Stack file not found
- **TypeError**: Wrong type for parameters

**Example:**
```python
try:
    video_from_stack(stack, preprocess, 'output.mp4', codec='h265')
except RuntimeError:
    print("H265 not available, falling back to H264")
    video_from_stack(stack, preprocess, 'output.mp4', codec='h264')
```

---

For more examples, see `examples.py` and `README.md`.
