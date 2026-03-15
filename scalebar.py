"""
Scalebar & Video-Speed Overlay
===============================

Shared helpers used by ``video_creator.create_video``,
``video_creator.create_velocity_video``, and
``create_three_stage_protocol_video``.

Place this file as ``fastvideo/scalebar.py``.

Author: Carlo
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# "Nice" bar lengths in µm, ordered small → large
_NICE_LENGTHS_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def pick_scalebar_length_um(
    image_width_px: int,
    pixel_size_um: float,
    target_fraction: float = 0.15,
) -> float:
    """
    Choose a round physical length whose bar is ≈ *target_fraction* of
    the image width.

    Parameters
    ----------
    image_width_px : int
        Width of the output image in pixels.
    pixel_size_um : float
        Effective µm / pixel at the output resolution.
    target_fraction : float
        Fraction of image width the bar should span (default 0.15).

    Returns
    -------
    float
        Round bar length in µm.
    """
    target_um = image_width_px * pixel_size_um * target_fraction
    best = _NICE_LENGTHS_UM[0]
    for L in _NICE_LENGTHS_UM:
        if abs(L - target_um) < abs(best - target_um):
            best = L
    return float(best)


def compute_video_speed(
    skip: int,
    real_dt: float,
    playback_fps: float,
) -> float:
    """
    Effective playback speed relative to real time.

    speed = skip × real_dt × playback_fps
    e.g.  skip=3, real_dt=1/30, fps=25 → 2.5×

    Parameters
    ----------
    skip : int
        Frame skip factor.
    real_dt : float
        Real inter-frame time step (seconds).
    playback_fps : float
        Playback frame rate.

    Returns
    -------
    float
        Speed multiplier.
    """
    return skip * real_dt * playback_fps


def format_speed(speed: float) -> str:
    """Format speed as a compact string, e.g. ``'2.5x'`` or ``'1x'``."""
    if abs(speed - round(speed)) < 0.05:
        return f"{int(round(speed))}x"
    return f"{speed:.1f}x"


def draw_scalebar_and_speed(
    img_rgb: np.ndarray,
    pixel_size_um: float,
    speed: float,
    scalebar_length_um: Optional[float] = None,
    margin: int = 20,
    bar_height: int = 6,
    bar_color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.6,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    Draw a spatial scalebar and video-speed label on the **top-right**
    corner of an image.

    Layout (right-aligned, top margin)::

        ┌──────────────────────────────────────────┐
        │                              ██████ 50 µm│
        │                                     2.5× │
        │                                          │

    Parameters
    ----------
    img_rgb : ndarray
        RGB image (H, W, 3) uint8.  Modified **in-place**.
    pixel_size_um : float
        Effective µm / pixel at the current output resolution.
    speed : float
        Playback speed relative to real time.
    scalebar_length_um : float or None
        Physical length of the bar in µm.  ``None`` → auto-pick.
    margin : int
        Pixel margin from the top-right corner.
    bar_height : int
        Bar thickness in pixels.
    bar_color : tuple
        RGB colour for the bar and text.
    outline_color : tuple
        RGB colour for text outline (contrast).
    font_scale : float
        cv2 font scale.
    font_thickness : int
        cv2 font thickness.

    Returns
    -------
    ndarray
        The same ``img_rgb`` array (modified in-place).
    """
    h, w = img_rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Auto-pick if not specified
    if scalebar_length_um is None:
        scalebar_length_um = pick_scalebar_length_um(w, pixel_size_um)

    bar_length_px = int(round(scalebar_length_um / pixel_size_um))

    # Labels
    bar_label = f"{scalebar_length_um:.0f} um"
    speed_label = format_speed(speed)

    # Measure text sizes
    (tw_bar, th_bar), _ = cv2.getTextSize(bar_label, font, font_scale, font_thickness)
    (tw_spd, th_spd), _ = cv2.getTextSize(speed_label, font, font_scale, font_thickness)

    # Layout anchors — right-aligned from the right edge
    right_edge = w - margin

    x_bar_right = right_edge - tw_bar - 8
    x_bar_left = x_bar_right - bar_length_px
    y_bar_top = margin
    y_bar_bot = margin + bar_height
    y_text_bar = y_bar_bot + th_bar + 4
    y_text_spd = y_text_bar + th_spd + 6

    # Bar outline + fill
    cv2.rectangle(
        img_rgb,
        (x_bar_left - 1, y_bar_top - 1),
        (x_bar_right + 1, y_bar_bot + 1),
        outline_color, -1, cv2.LINE_AA,
    )
    cv2.rectangle(
        img_rgb,
        (x_bar_left, y_bar_top),
        (x_bar_right, y_bar_bot),
        bar_color, -1, cv2.LINE_AA,
    )

    # Bar label (with outline for contrast)
    x_text_bar = right_edge - tw_bar
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        cv2.putText(
            img_rgb, bar_label,
            (x_text_bar + dx, y_text_bar + dy),
            font, font_scale, outline_color,
            font_thickness + 1, cv2.LINE_AA,
        )
    cv2.putText(
        img_rgb, bar_label,
        (x_text_bar, y_text_bar),
        font, font_scale, bar_color,
        font_thickness, cv2.LINE_AA,
    )

    # Speed label (with outline for contrast)
    x_text_spd = right_edge - tw_spd
    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        cv2.putText(
            img_rgb, speed_label,
            (x_text_spd + dx, y_text_spd + dy),
            font, font_scale, outline_color,
            font_thickness + 1, cv2.LINE_AA,
        )
    cv2.putText(
        img_rgb, speed_label,
        (x_text_spd, y_text_spd),
        font, font_scale, bar_color,
        font_thickness, cv2.LINE_AA,
    )

    return img_rgb




def draw_scalebar_inline(
    img_rgb: np.ndarray,
    pixel_size_um: float,
    speed: float,
    scalebar_length_um: Optional[float] = None,
    margin: int = 20,
    bar_color: Tuple[int, int, int] = (255, 255, 255),
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.55,
    font_thickness: int = 1,
    padding_x: int = 8,
    padding_y: int = 6,
) -> np.ndarray:
    """
    Draw a scalebar with size and speed labels **inside** the bar.

    Layout (right-aligned, top margin)::

        ┌──────────────────────────────┐
        │  50 µm               2.5×   │
        └──────────────────────────────┘

    The bar width is determined by the physical scale (in pixels).
    If the text is wider than the physical bar, the bar is enlarged
    to fit the labels.  The bar height auto-sizes to the text.

    Parameters
    ----------
    img_rgb : ndarray
        RGB image (H, W, 3) uint8.  Modified **in-place**.
    pixel_size_um : float
        Effective µm / pixel at the current output resolution.
    speed : float
        Playback speed relative to real time.
    scalebar_length_um : float or None
        Physical bar length in µm.  ``None`` → auto-pick.
    margin : int
        Pixel margin from the top-right corner.
    bar_color : tuple
        RGB fill colour of the bar rectangle (also used for text).
    outline_color : tuple
        RGB colour for the dark outline / text shadow.
    font_scale : float
        cv2 font scale.
    font_thickness : int
        cv2 font thickness.
    padding_x : int
        Horizontal padding inside the bar (pixels).
    padding_y : int
        Vertical padding inside the bar (pixels).

    Returns
    -------
    ndarray
        Same ``img_rgb`` (modified in-place).
    """
    h, w = img_rgb.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Auto-pick bar length
    if scalebar_length_um is None:
        scalebar_length_um = pick_scalebar_length_um(w, pixel_size_um)

    bar_length_px = int(round(scalebar_length_um / pixel_size_um))
    bar_length_px = max(bar_length_px, 4)

    # Format labels
    if scalebar_length_um >= 1000:
        size_label = f'{scalebar_length_um / 1000:.3g} mm'
    else:
        size_label = f'{int(scalebar_length_um)} um'

    speed_label = format_speed(speed)

    # Measure text
    (tw_size, th_size), _ = cv2.getTextSize(size_label, font, font_scale, font_thickness)
    (tw_spd, th_spd), _ = cv2.getTextSize(speed_label, font, font_scale, font_thickness)

    text_height = max(th_size, th_spd)
    min_text_width = tw_size + tw_spd + 3 * padding_x  # left_pad + gap + right_pad

    # Bar dimensions: width from physics, but at least wide enough for text
    bar_width = max(bar_length_px, min_text_width)
    bar_height = text_height + 2 * padding_y

    # Position: top-right corner
    x_bar_right = w - margin
    x_bar_left = x_bar_right - bar_width
    y_bar_top = margin
    y_bar_bot = y_bar_top + bar_height

    # --- Draw dark outline rectangle ---
    cv2.rectangle(
        img_rgb,
        (x_bar_left - 1, y_bar_top - 1),
        (x_bar_right + 1, y_bar_bot + 1),
        outline_color, -1, cv2.LINE_AA,
    )

    # --- Draw filled bar ---
    cv2.rectangle(
        img_rgb,
        (x_bar_left, y_bar_top),
        (x_bar_right, y_bar_bot),
        bar_color, -1, cv2.LINE_AA,
    )

    # --- Text colour: choose contrast (black or white) against the bar ---
    # Perceived brightness of bar_color
    brightness = 0.299 * bar_color[0] + 0.587 * bar_color[1] + 0.114 * bar_color[2]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    shadow_color = (255, 255, 255) if brightness > 128 else (0, 0, 0)

    # Text baseline: vertically centered inside bar
    y_text = y_bar_top + padding_y + text_height

    # Size label — left-aligned inside bar
    x_text_size = x_bar_left + padding_x

    # Speed label — right-aligned inside bar
    x_text_spd = x_bar_right - padding_x - tw_spd

    # Draw text with thin shadow for readability
    for label, x_pos in [(size_label, x_text_size), (speed_label, x_text_spd)]:
        # Shadow (1 px offset)
        cv2.putText(
            img_rgb, label, (x_pos + 1, y_text + 1),
            font, font_scale, shadow_color,
            font_thickness, cv2.LINE_AA,
        )
        # Main text
        cv2.putText(
            img_rgb, label, (x_pos, y_text),
            font, font_scale, text_color,
            font_thickness, cv2.LINE_AA,
        )

    return img_rgb
