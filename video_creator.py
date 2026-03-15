"""
Video Creation Functions
========================

Main functions for creating velocity field videos with validation
and error handling.

Updated: scalebar + video-speed overlay in ``create_video`` and
``create_velocity_video``, using shared helpers from ``scalebar.py``.

Author: Carlo
"""

import numpy as np
import cv2
import tqdm
from pathlib import Path
from typing import Callable, Optional, Tuple

from .ffmpeg_writer import RobustFFmpegWriter, check_ffmpeg_codecs
from .overlay import VelocityOverlayProcessor, GPUVelocityOverlayProcessor
from .utils import calculate_optimal_resolution, ensure_even_dimensions
from .preview import validate_video_settings
from .scalebar import (
    pick_scalebar_length_um,
    compute_video_speed,
    format_speed,
    draw_scalebar_and_speed,
    draw_scalebar_inline
)


# ============================================================================
# create_video  (plain — no arrows, no stages)
# ============================================================================

def create_video(
    stack,
    preprocessor: Callable,
    output_path: str,
    pixel_size: float = 0.65,
    target_size_mb: float = 200,
    dt: Optional[float] = None,
    fps: float = 25.0,
    codec: str = "h264",
    preset: str = "medium",
    resolution_scale: Optional[float] = None,
    start: int = 0,
    end: Optional[int] = None,
    skip: int = 1,
    text: bool = False,
    text_position: Tuple[int, int] = (50, 50),
    text_size: float = 1.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    auto_optimize_resolution: bool = True,
    use_gpu: bool = False,
    verbose: bool = False,
    # --- scalebar / speed ---
    show_scalebar: bool = True,
    scalebar_length_um: Optional[float] = None,
    scalebar_margin: int = 20,
    scalebar_bar_height: int = 6,
    scalebar_color: Tuple[int, int, int] = (255, 255, 255),
    scalebar_font_scale: float = 0.6,
):
    """
    Create a video from preprocessed microscopy frames with an optional
    scalebar and video-speed indicator.

    This is the **plain** video creator — no velocity arrows, no seeder
    trajectories.  Just image preprocessing + encoding.

    Parameters
    ----------
    stack : object
        Image stack (``stack[i].data``, ``stack.time()``).
    preprocessor : callable
        ``preprocess(img) -> normalised_img``.
    output_path : str
        Output ``.mp4`` file path.
    pixel_size : float
        µm / pixel at the original acquisition resolution.
    target_size_mb : float
        Target file size in MB.
    dt : float or None
        Time step in seconds.  ``None`` → derived from ``stack.time()``.
    fps : float
        Playback frame rate.
    codec, preset : str
        FFmpeg codec / preset.
    resolution_scale : float or None
        Manual downscaling factor.
    start, end, skip : int
        Frame range and skip factor.
    text : bool
        Overlay timestamp.
    text_position, text_size, text_color
        Timestamp style.
    auto_optimize_resolution : bool
        Automatically compute resolution to hit ``target_size_mb``.
    use_gpu : bool
        Reserved for future use.
    verbose : bool
        Print FFmpeg output.
    show_scalebar : bool
        Draw a scalebar and video-speed indicator on the top-right
        corner.  Speed is computed from ``skip``, real acquisition dt,
        and ``fps``.
    scalebar_length_um : float or None
        Physical length of the bar in µm.  ``None`` → auto-pick.
    scalebar_margin : int
        Pixel margin from the top-right corner.
    scalebar_bar_height : int
        Bar thickness in pixels.
    scalebar_color : tuple
        RGB colour for bar and labels.
    scalebar_font_scale : float
        cv2 font scale for scalebar / speed text.

    Examples
    --------
    >>> create_video(
    ...     stack, preprocessor,
    ...     output_path='bacteria.mp4',
    ...     pixel_size=0.65,
    ...     skip=3,
    ...     show_scalebar=True,
    ... )
    """
    # ------------------------------------------------------------------
    # Time / frame range
    # ------------------------------------------------------------------
    times = stack.time()
    if dt is None:
        dt = float(np.mean(np.diff(times[1:])))

    if end is None or end <= 0:
        end = len(stack)
    end = min(end, len(stack))

    n_frames = (end - start) // skip
    duration = n_frames / fps

    # ------------------------------------------------------------------
    # Original dimensions
    # ------------------------------------------------------------------
    first_img = preprocessor(stack[0].data.astype(float))
    try:
        if hasattr(first_img, "get"):
            first_img = first_img.get()
    except:
        pass
    first_img = np.asarray(first_img)
    original_height, original_width = first_img.shape[:2]

    # ------------------------------------------------------------------
    # Output resolution
    # ------------------------------------------------------------------
    if auto_optimize_resolution and resolution_scale is None:
        opt_width, opt_height, bitrate = calculate_optimal_resolution(
            original_width, original_height,
            target_size_mb, duration, fps, codec,
        )
        output_resolution = (opt_width, opt_height)
    elif resolution_scale is not None:
        opt_width = int(original_width * resolution_scale)
        opt_height = int(original_height * resolution_scale)
        opt_width, opt_height = ensure_even_dimensions(opt_width, opt_height)
        output_resolution = (opt_width, opt_height)
        bitrate_kbps = int(target_size_mb * 8 * 1024 / max(duration, 1))
        bitrate = f"{bitrate_kbps}k"
    else:
        opt_width, opt_height = ensure_even_dimensions(
            original_width, original_height,
        )
        output_resolution = (
            None
            if (opt_width, opt_height) == (original_width, original_height)
            else (opt_width, opt_height)
        )
        bitrate = "500k"

    # ------------------------------------------------------------------
    # Scalebar & speed
    # ------------------------------------------------------------------
    pixel_size_out = pixel_size * (original_width / opt_width)
    dt_real = float(np.median(np.diff(times[: min(100, len(times))])))
    video_speed = compute_video_speed(skip, dt_real, fps)

    if show_scalebar:
        _scalebar_len = (
            scalebar_length_um
            if scalebar_length_um is not None
            else pick_scalebar_length_um(opt_width, pixel_size_out)
        )
    else:
        _scalebar_len = None

    # ------------------------------------------------------------------
    # Time array
    # ------------------------------------------------------------------
    if text:
        time_array = times[start:end:skip]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VIDEO CREATION")
    print("=" * 70)
    print(f"  Processing       : {'GPU' if use_gpu else 'CPU'}")
    print(f"  Frames           : {start} to {end}, skip={skip} ({n_frames} total)")
    print(f"  Duration         : {duration:.1f} s")
    print(f"  Original         : {original_width}×{original_height}")
    print(f"  Output           : {opt_width}×{opt_height}")
    print(f"  Target size      : {target_size_mb} MB")
    print(f"  Codec / preset   : {codec} / {preset}")
    print(f"  Bitrate          : {bitrate}")
    if show_scalebar:
        print(f"  Scalebar         : {_scalebar_len:.0f} µm  "
              f"({_scalebar_len / pixel_size_out:.0f} px)")
        print(f"  Video speed      : {format_speed(video_speed)}  "
              f"(skip={skip}, dt_real={dt_real*1000:.1f} ms, fps={fps})")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    with RobustFFmpegWriter(
        output_path, opt_width, opt_height, fps,
        bitrate=bitrate, codec=codec, preset=preset,
        verbose=verbose,
    ) as writer:

        for i, frame_idx in enumerate(
            tqdm.tqdm(range(start, end, skip), desc="Rendering video")
        ):
            img = preprocessor(stack[frame_idx].data.astype(float))

            # Handle CuPy arrays
            try:
                if hasattr(img, "get"):
                    img = img.get()
            except:
                pass
            img = np.asarray(img)

            # Normalize to uint8
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Grayscale → RGB (needed for coloured overlays)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            # Resize to output resolution
            if img.shape[:2] != (opt_height, opt_width):
                img = cv2.resize(
                    img, (opt_width, opt_height),
                    interpolation=cv2.INTER_AREA,
                )
                img = np.ascontiguousarray(img)

            # Timestamp
            if text:
                t_label = f"t: {time_array[i]:.2f}s"
                cv2.putText(
                    img, t_label, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    text_color, 2, cv2.LINE_AA,
                )

            # Scalebar + speed (top-right)
            if show_scalebar:
                draw_scalebar_and_speed(
                    img,
                    pixel_size_um=pixel_size_out,
                    speed=video_speed,
                    scalebar_length_um=_scalebar_len,
                    margin=scalebar_margin,
                    bar_height=scalebar_bar_height,
                    bar_color=scalebar_color,
                    font_scale=scalebar_font_scale,
                )

            writer.write(img)

    print("\n✓ Video creation complete!")


# ============================================================================
# create_velocity_video  (with arrow overlay)
# ============================================================================

def create_velocity_video(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_path: str,
    target_size_mb: float = 200,
    dt: Optional[float] = None,
    fps: float = 25.0,
    codec: str = "h264",
    preset: str = "medium",
    resolution_scale: Optional[float] = None,
    arrow_scale: Optional[float] = None,
    arrow_width: int = 2,
    arrow_color: Tuple[int, int, int] = (255, 255, 0),
    subsample: Optional[int] = None,
    velocity_skip: Optional[int] = None,
    bitratefix: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    skip: int = 1,
    text: bool = False,
    text_position: Tuple[int, int] = (50, 50),
    text_size: float = 1.5,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    auto_optimize_resolution: bool = True,
    use_gpu: bool = False,
    verbose: bool = False,
    # --- scalebar / speed ---
    show_scalebar: bool = True,
    scalebar_isinline : bool = True,
    scalebar_length_um: Optional[float] = None,
    scalebar_margin: int = 20,
    scalebar_bar_height: int = 6,
    scalebar_color: Tuple[int, int, int] = (255, 255, 255),
    scalebar_font_scale: float = 0.6,
):
    """
    Create ultra-compressed video with velocity field overlay,
    scalebar, and video-speed indicator.

    Parameters
    ----------
    stack : object
        Image stack (``stack[i].data``, ``stack.time()``).
    preprocessor : callable
        ``preprocess(img) -> normalised_img``.
    x, y : ndarray
        Grid coordinates in physical units (µm), shape ``[ny, nx]``.
    u, v : ndarray
        Velocity in physical units (µm/s).
        ``[nt, ny, nx]`` for time series or ``[ny, nx]`` for single frame.
    pixel_size : float
        µm / pixel at the original acquisition resolution.
    output_path : str
        Output ``.mp4`` file path.
    target_size_mb : float
        Target file size in MB.
    dt : float or None
        Time step in seconds.  ``None`` → derived from ``stack.time()``.
    fps : float
        Playback frame rate.
    codec, preset : str
        FFmpeg codec / preset.
    resolution_scale : float or None
        Manual downscaling factor.
    arrow_scale, arrow_width, arrow_color, subsample
        Arrow rendering parameters (see ``VelocityOverlayProcessor``).
    velocity_skip : int or None
        Ratio of image frames to velocity frames.  If the velocity
        field was computed every Nth image frame, set this to N.
        ``None`` → auto-detect from ``len(stack) / u.shape[0]``.
    bitratefix : str or None
        Fix bitrate for the video creation
    start, end, skip : int
        Frame range and skip factor.
    text : bool
        Overlay timestamp.
    text_position, text_size, text_color
        Timestamp style.
    auto_optimize_resolution : bool
        Automatically compute resolution to hit ``target_size_mb``.
    use_gpu : bool
        Use GPU-accelerated overlay processor.
    verbose : bool
        Print FFmpeg output.
    show_scalebar : bool
        Draw a scalebar and video-speed indicator on the top-right
        corner.  Speed is computed from ``skip``, real acquisition dt,
        and ``fps``.
    scalebar_isinline : bool
        type of scalebar to be used
    scalebar_length_um : float or None
        Physical length of the bar in µm.  ``None`` → auto-pick.
    scalebar_margin : int
        Pixel margin from the top-right corner.
    scalebar_bar_height : int
        Bar thickness in pixels.
    scalebar_color : tuple
        RGB colour for bar and labels.
    scalebar_font_scale : float
        cv2 font scale for scalebar / speed text.

    Examples
    --------
    >>> create_velocity_video(
    ...     stack, preprocessor, x, y, u, v,
    ...     pixel_size=0.65,
    ...     output_path='velocity.mp4',
    ...     skip=3,
    ...     show_scalebar=True,
    ... )
    """
    # ------------------------------------------------------------------
    # Time / frame range
    # ------------------------------------------------------------------
    times = stack.time()
    if dt is None:
        dt = float(np.mean(np.diff(times[1:])))

    if end is None or end <= 0:
        end = len(stack)
    end = min(end, len(stack))

    n_frames = (end - start) // skip
    duration = n_frames / fps

    # ------------------------------------------------------------------
    # Velocity skip factor  (image frames per velocity frame)
    # ------------------------------------------------------------------
    n_vel_frames = u.shape[0] if u.ndim == 3 else 1
    n_stack_frames = len(stack)

    if velocity_skip is None:
        if u.ndim == 3 and n_vel_frames < n_stack_frames:
            velocity_skip = max(1, round(n_stack_frames / n_vel_frames))
            print(f"Auto-detected velocity_skip={velocity_skip}  "
                  f"({n_stack_frames} images / {n_vel_frames} velocity frames)")
        else:
            velocity_skip = 1
    else:
        velocity_skip = max(1, int(velocity_skip))

    # ------------------------------------------------------------------
    # Original dimensions
    # ------------------------------------------------------------------
    first_img = preprocessor(stack[0].data.astype(float))
    if hasattr(first_img, "get"):
        first_img = first_img.get()
    first_img = np.asarray(first_img)
    original_height, original_width = first_img.shape[:2]

    # ------------------------------------------------------------------
    # Output resolution
    # ------------------------------------------------------------------
    if auto_optimize_resolution and resolution_scale is None:
        opt_width, opt_height, bitrate = calculate_optimal_resolution(
            original_width, original_height,
            target_size_mb, duration, fps, codec,
        )
        output_resolution = (opt_width, opt_height)
    elif resolution_scale is not None:
        opt_width = int(original_width * resolution_scale)
        opt_height = int(original_height * resolution_scale)
        opt_width, opt_height = ensure_even_dimensions(opt_width, opt_height)
        output_resolution = (opt_width, opt_height)
        bitrate_kbps = int(target_size_mb * 8 * 1024 / max(duration, 1))
        bitrate = f"{bitrate_kbps}k"
    else:
        opt_width, opt_height = ensure_even_dimensions(
            original_width, original_height,
        )
        output_resolution = (
            None
            if (opt_width, opt_height) == (original_width, original_height)
            else (opt_width, opt_height)
        )
        bitrate = "500k"
        
    if bitratefix is not None:
        bitrate = bitratefix

    # ------------------------------------------------------------------
    # Scalebar & speed
    # ------------------------------------------------------------------
    pixel_size_out = pixel_size * (original_width / opt_width)
    dt_real = float(np.median(np.diff(times[: min(100, len(times))])))
    video_speed = compute_video_speed(skip, dt_real, fps)

    if show_scalebar:
        _scalebar_len = (
            scalebar_length_um
            if scalebar_length_um is not None
            else pick_scalebar_length_um(opt_width, pixel_size_out)
        )
    else:
        _scalebar_len = None

    # ------------------------------------------------------------------
    # Overlay processor
    # ------------------------------------------------------------------
    if use_gpu:
        try:
            processor = GPUVelocityOverlayProcessor(
                preprocessor, x, y, u, v, pixel_size, dt,
                output_resolution=None,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                arrow_color=arrow_color,
                subsample=subsample,
            )
            print("Using GPU acceleration")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU processing")
            use_gpu = False
            processor = VelocityOverlayProcessor(
                preprocessor, x, y, u, v, pixel_size, dt,
                output_resolution=None,
                arrow_scale=arrow_scale,
                arrow_width=arrow_width,
                arrow_color=arrow_color,
                subsample=subsample,
            )
    else:
        processor = VelocityOverlayProcessor(
            preprocessor, x, y, u, v, pixel_size, dt,
            output_resolution=None,
            arrow_scale=arrow_scale,
            arrow_width=arrow_width,
            arrow_color=arrow_color,
            subsample=subsample,
        )

    # ------------------------------------------------------------------
    # Time array
    # ------------------------------------------------------------------
    if text:
        time_array = times[start:end:skip]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VELOCITY VIDEO CREATION")
    print("=" * 70)
    print(f"  Processing       : {'GPU' if use_gpu else 'CPU'}")
    print(f"  Frames           : {start} to {end}, skip={skip} ({n_frames} total)")
    print(f"  Velocity skip    : {velocity_skip}  "
          f"({n_vel_frames} vel frames / {n_stack_frames} images)")
    print(f"  Duration         : {duration:.1f} s")
    print(f"  Original         : {original_width}×{original_height}")
    print(f"  Output           : {opt_width}×{opt_height}")
    print(f"  Target size      : {target_size_mb} MB")
    print(f"  Codec / preset   : {codec} / {preset}")
    print(f"  Bitrate          : {bitrate}")
    if show_scalebar:
        print(f"  Scalebar         : {_scalebar_len:.0f} µm  "
              f"({_scalebar_len / pixel_size_out:.0f} px)")
        print(f"  Video speed      : {format_speed(video_speed)}  "
              f"(skip={skip}, dt_real={dt_real*1000:.1f} ms, fps={fps})")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    with RobustFFmpegWriter(
        output_path, opt_width, opt_height, fps,
        bitrate=bitrate, codec=codec, preset=preset,
        verbose=verbose,
    ) as writer:

        for i, frame_idx in enumerate(
            tqdm.tqdm(range(start, end, skip), desc="Rendering velocity video")
        ):
            # Map image frame index → velocity frame index
            vel_idx = frame_idx // velocity_skip

            img_with_arrows = processor(
                stack[frame_idx].data.astype(float),
                vel_idx,
            )

            # Ensure correct dimensions
            if img_with_arrows.shape[:2] != (opt_height, opt_width):
                img_with_arrows = cv2.resize(
                    img_with_arrows, (opt_width, opt_height),
                    interpolation=cv2.INTER_AREA,
                )
                img_with_arrows = np.ascontiguousarray(img_with_arrows)

            # Timestamp
            if text:
                t_label = f"t: {time_array[i]:.2f}s"
                cv2.putText(
                    img_with_arrows, t_label, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    text_color, 2, cv2.LINE_AA,
                )

            # Scalebar + speed (top-right)
            if show_scalebar:
                if scalebar_isinline:
                    draw_scalebar_inline(
                        img_with_arrows,
                        pixel_size_um=pixel_size_out,
                        speed=video_speed,
                        scalebar_length_um=_scalebar_len,
                        margin=scalebar_margin,
                        bar_color=scalebar_color,
                        font_scale=scalebar_font_scale,
                    )
                else:
                    draw_scalebar_and_speed(
                        img_with_arrows,
                        pixel_size_um=pixel_size_out,
                        speed=video_speed,
                        scalebar_length_um=_scalebar_len,
                        margin=scalebar_margin,
                        bar_height=scalebar_bar_height,
                        bar_color=scalebar_color,
                        font_scale=scalebar_font_scale,
                    )

            writer.write(img_with_arrows)

    print("\n✓ Velocity video creation complete!")


# ============================================================================
# Convenience wrappers (unchanged except they pass through new args)
# ============================================================================

def create_video_simple(
    stack,
    preprocessor: Callable,
    velocity_file: str,
    output_path: str,
    pixel_size: float = 0.65,
    target_size_mb: float = 200,
    dt: float = 1 / 25,
    fps: float = 25,
    skip: int = 1,
    text: bool = True,
    use_gpu: bool = False,
    show_scalebar: bool = True,
):
    """
    Simplified interface for common use case.

    Loads velocity data from ``.npz`` file and creates video with
    sensible defaults.
    """
    vel = np.load(velocity_file)

    try:
        create_velocity_video(
            stack, preprocessor,
            vel["x"], vel["y"], vel["u"], vel["v"],
            dt=dt,
            pixel_size=pixel_size,
            output_path=output_path,
            target_size_mb=target_size_mb,
            fps=fps,
            codec="h265_nvenc" if use_gpu else "h265",
            skip=skip,
            text=text,
            use_gpu=use_gpu,
            show_scalebar=show_scalebar,
        )
    except Exception as e:
        if "nvenc" in str(e).lower() or "gpu" in str(e).lower():
            print(f"\nGPU encoding failed: {e}")
            print("Retrying with CPU encoding...")
            create_velocity_video(
                stack, preprocessor,
                vel["x"], vel["y"], vel["u"], vel["v"],
                dt=dt,
                pixel_size=pixel_size,
                output_path=output_path,
                target_size_mb=target_size_mb,
                fps=fps,
                codec="h264",
                skip=skip,
                text=text,
                use_gpu=False,
                show_scalebar=show_scalebar,
            )
        else:
            raise


def create_video_with_validation(
    stack,
    preprocessor: Callable,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pixel_size: float,
    output_path: str,
    target_size_mb: float = 200,
    dt: float = 1 / 25,
    fps: float = 25.0,
    codec: str = "h264",
    **kwargs,
) -> Optional[dict]:
    """
    Complete workflow: validate → preview → create video.
    """
    from .preview import validate_video_settings

    report = validate_video_settings(
        stack, preprocessor, x, y, u, v,
        pixel_size=pixel_size,
        output_path=output_path,
        target_size_mb=target_size_mb,
        fps=fps,
        codec=codec,
    )

    if report["is_valid"]:
        create_velocity_video(
            stack, preprocessor,
            x, y, u, v,
            pixel_size=pixel_size,
            output_path=output_path,
            target_size_mb=target_size_mb,
            dt=dt,
            fps=fps,
            codec=codec,
            **kwargs,
        )

    return report