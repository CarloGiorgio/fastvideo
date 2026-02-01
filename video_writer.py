"""
Video Writer with Codec Management
===================================

Low-level video writer with automatic codec selection and quality control.
Provides a clean interface to OpenCV's VideoWriter with robust error handling.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings


class VideoWriter:
    """
    Optimized video writer with automatic codec selection and quality control.
    
    This class wraps OpenCV's VideoWriter with enhanced features:
    - Automatic codec fallback if primary codec fails
    - Quality presets (high/medium/low) with appropriate bitrate settings
    - Proper handling of color vs grayscale videos
    - Context manager support for safe resource handling
    - Validation of video parameters
    
    Parameters
    ----------
    filename : str
        Output video filename (must end with .mp4, .avi, etc.)
    width : int
        Frame width in pixels
    height : int
        Frame height in pixels
    fps : float
        Frames per second
    is_color : bool, optional
        Whether video is color (True) or grayscale (False), default True
    codec : str, optional
        Video codec: 'h264', 'h265', 'mp4v', 'xvid', 'mjpg', default 'h264'
    quality : str, optional
        Quality preset: 'high', 'medium', 'low', default 'medium'
    
    Examples
    --------
    >>> # Using context manager (recommended)
    >>> with VideoWriter('output.mp4', 1920, 1080, 30.0, is_color=True) as writer:
    ...     for frame in frames:
    ...         writer.write(frame)
    
    >>> # Manual usage
    >>> writer = VideoWriter('output.mp4', 1920, 1080, 30.0)
    >>> writer.write(frame)
    >>> writer.release()
    
    Notes
    -----
    - For grayscale videos, ensure frames are 2D arrays (height, width)
    - For color videos, ensure frames are 3D arrays (height, width, 3) in BGR
    - The writer validates frame dimensions on each write call
    - If the primary codec fails, falls back to 'mp4v' automatically
    """
    
    # Codec mappings with fallback options
    CODEC_MAP = {
        'h264': ['H264', 'h264', 'avc1', 'X264'],
        'h265': ['HEVC', 'hev1', 'X265', 'H265'],
        'mp4v': ['mp4v', 'MP4V', 'FMP4'],
        'xvid': ['XVID', 'xvid'],
        'mjpg': ['MJPG', 'mjpg', 'MJPEG']
    }
    
    # Quality presets - these affect compression level
    QUALITY_PARAMS = {
        'high': {'crf': 18, 'preset': 'slow'},
        'medium': {'crf': 23, 'preset': 'medium'},
        'low': {'crf': 28, 'preset': 'fast'}
    }
    
    def __init__(self, filename: str, width: int, height: int, fps: float,
                 is_color: bool = True, codec: str = 'h264', quality: str = 'medium'):
        
        # Validate inputs
        if width <= 0 or height <= 0:
            raise ValueError(f"Width and height must be positive, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        
        # Ensure even dimensions (required by some codecs)
        if width % 2 != 0:
            width += 1
            warnings.warn(f"Width must be even, adjusted to {width}")
        if height % 2 != 0:
            height += 1
            warnings.warn(f"Height must be even, adjusted to {height}")
        
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.is_color = is_color
        self.codec_name = codec
        self.quality = quality
        self.writer = None
        self.frame_count = 0
        
        # Initialize writer
        self._init_writer()
    
    def _init_writer(self):
        """Initialize the OpenCV VideoWriter with codec fallback."""
        # Try primary codec options
        codec_list = self.CODEC_MAP.get(self.codec_name.lower(), 
                                       ['mp4v'])  # Fallback to mp4v
        
        for fourcc_str in codec_list:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(
                    self.filename,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    self.is_color
                )
                
                if writer.isOpened():
                    self.writer = writer
                    self.actual_codec = fourcc_str
                    return
                else:
                    writer.release()
            
            except Exception as e:
                continue
        
        # If all failed, try generic mp4v as last resort
        if self.writer is None or not self.writer.isOpened():
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    self.filename,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                    self.is_color
                )
                if writer.isOpened():
                    self.writer = writer
                    self.actual_codec = 'mp4v'
                    warnings.warn(
                        f"Requested codec '{self.codec_name}' not available, "
                        f"falling back to 'mp4v'"
                    )
                else:
                    raise RuntimeError(
                        f"Failed to initialize video writer. "
                        f"No compatible codec found for '{self.filename}'"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize video writer: {e}\n"
                    f"Requested: {self.codec_name}, tried fallbacks"
                )
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame to write. For color videos, shape should be (height, width, 3)
            in BGR format. For grayscale, shape should be (height, width).
        
        Raises
        ------
        ValueError
            If frame dimensions don't match writer dimensions
        RuntimeError
            If writer is not initialized
        """
        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError("VideoWriter is not initialized")
        
        # Validate frame dimensions
        if self.is_color:
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Expected color frame with shape (height, width, 3), "
                    f"got shape {frame.shape}"
                )
            if frame.shape[:2] != (self.height, self.width):
                raise ValueError(
                    f"Frame dimensions {frame.shape[:2]} don't match "
                    f"writer dimensions ({self.height}, {self.width})"
                )
        else:
            if frame.ndim != 2:
                # If 3D grayscale, convert to 2D
                if frame.ndim == 3 and frame.shape[2] == 1:
                    frame = frame[:, :, 0]
                else:
                    raise ValueError(
                        f"Expected grayscale frame with shape (height, width), "
                        f"got shape {frame.shape}"
                    )
            if frame.shape != (self.height, self.width):
                raise ValueError(
                    f"Frame dimensions {frame.shape} don't match "
                    f"writer dimensions ({self.height}, {self.width})"
                )
        
        # Ensure correct dtype
        if frame.dtype != np.uint8:
            # Normalize to [0, 255] if float
            if frame.dtype in [np.float32, np.float64]:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release the video writer and close the file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures writer is released."""
        self.release()
        return False
    
    def __del__(self):
        """Destructor - ensures writer is released."""
        self.release()
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the video being written.
        
        Returns
        -------
        dict
            Dictionary with video statistics
        """
        return {
            'filename': self.filename,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'is_color': self.is_color,
            'codec': self.actual_codec if hasattr(self, 'actual_codec') else self.codec_name,
            'quality': self.quality,
            'frames_written': self.frame_count,
            'duration_seconds': self.frame_count / self.fps if self.fps > 0 else 0
        }


def check_available_codecs(verbose: bool = True) -> List[str]:
    """
    Test which video codecs are available on the system.
    
    This function tests common video codecs by attempting to create
    a temporary video file with each codec. Useful for debugging
    codec availability issues.
    
    Parameters
    ----------
    verbose : bool, optional
        If True, print detailed information. If False, just return list.
        Default True.
    
    Returns
    -------
    list of str
        List of available codec names
    
    Examples
    --------
    >>> available = check_available_codecs()
    Testing available video codecs:
    --------------------------------------------------
    ✓ H264 (recommended): use 'h264'
    ✓ H265 (high compression): use 'h265'
    ✓ MP4V (compatible): use 'mp4v'
    ...
    
    >>> # Get list without printing
    >>> codecs = check_available_codecs(verbose=False)
    >>> print(codecs)
    ['h264', 'h265', 'mp4v']
    """
    if verbose:
        print("Testing available video codecs:")
        print("-" * 50)
    
    test_codecs = {
        'H264 (recommended)': (['H264', 'h264', 'avc1', 'X264'], 'h264'),
        'H265 (high compression)': (['HEVC', 'hev1', 'X265'], 'h265'),
        'MP4V (compatible)': (['mp4v', 'MP4V'], 'mp4v'),
        'XVID (alternative)': (['XVID', 'xvid'], 'xvid'),
        'MJPEG (fast)': (['MJPG', 'mjpg'], 'mjpg')
    }
    
    available = []
    
    for codec_name, (fourcc_list, codec_key) in test_codecs.items():
        working = False
        for fourcc_str in fourcc_list:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter('test.mp4', fourcc, 30.0, (640, 480), True)
                if out.isOpened():
                    out.release()
                    if verbose:
                        print(f"✓ {codec_name}: use '{codec_key}'")
                    available.append(codec_key)
                    working = True
                    break
            except:
                continue
        
        if not working and verbose:
            print(f"✗ {codec_name}: not available")
    
    # Clean up test file
    test_file = Path('test.mp4')
    if test_file.exists():
        test_file.unlink()
    
    if verbose:
        print("-" * 50)
        if available:
            print(f"Recommended codec: '{available[0]}'")
        else:
            print("⚠ Warning: No H264/H265 codecs found.")
            print("  Install ffmpeg with codec support for best results.")
    
    return available


# Backward compatibility alias
VideoWriterOptimized = VideoWriter
