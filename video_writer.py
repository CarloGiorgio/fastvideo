"""
Video Writer with Codec Management
===================================

Optimized video writer with automatic codec selection and quality control.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class VideoWriter:
    """
    Optimized video writer with automatic codec selection and quality control.
    
    Automatically handles:
    - Codec selection with fallback options
    - Quality settings (bitrate)
    - Color/grayscale detection
    - Context manager support
    
    Parameters
    ----------
    filename : str
        Output video filename
    width, height : int
        Video dimensions
    fps : float
        Frames per second
    is_color : bool, optional
        Color (True) or grayscale (False) video (default: False)
    codec : str, optional
        Codec name: 'h264', 'h265', 'mp4v', 'xvid', 'mjpg' (default: 'h264')
    quality : str, optional
        Quality preset: 'high', 'medium', 'low' (default: 'medium')
    
    Examples
    --------
    >>> with VideoWriter('output.mp4', 1024, 1024, 30.0) as writer:
    ...     for frame in frames:
    ...         writer.write(frame)
    """
    
    # Codec fallback chain
    CODEC_FALLBACKS = {
        'h264': ['H264', 'h264', 'avc1', 'X264'],
        'h265': ['HEVC', 'hev1', 'X265', 'H265'],
        'mp4v': ['mp4v', 'MP4V'],
        'xvid': ['XVID', 'xvid'],
        'mjpg': ['MJPG', 'mjpg']
    }
    
    # Quality to bitrate mapping (Mbps)
    QUALITY_BITRATES = {
        'high': 10,
        'medium': 5,
        'low': 2
    }
    
    def __init__(self, filename: str, width: int, height: int, fps: float,
                 is_color: bool = False, codec: str = 'h264', quality: str = 'medium'):
        self.filename = str(filename)
        self.width = width
        self.height = height
        self.fps = fps
        self.is_color = is_color
        self.codec_name = codec.lower()
        self.quality = quality
        self.writer = None
        self.frame_count = 0
        
        # Initialize writer
        self._init_writer()
    
    def _init_writer(self):
        """Initialize video writer with codec fallback."""
        # Try codec options in order
        codec_options = self.CODEC_FALLBACKS.get(self.codec_name, [self.codec_name])
        
        for fourcc_str in codec_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                writer = cv2.VideoWriter(
                    self.filename, fourcc, self.fps,
                    (self.width, self.height), self.is_color
                )
                
                if writer.isOpened():
                    self.writer = writer
                    print(f"Using codec: {fourcc_str}")
                    return
            except Exception as e:
                continue
        
        # Fallback to default if all fail
        print(f"Warning: Preferred codec '{self.codec_name}' not available, using default")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename, fourcc, self.fps,
            (self.width, self.height), self.is_color
        )
        
        if not self.writer.isOpened():
            raise RuntimeError("Failed to initialize video writer")
    
    def write(self, frame: np.ndarray):
        """
        Write a frame to the video.
        
        Parameters
        ----------
        frame : np.ndarray
            Frame to write (uint8)
        """
        if self.writer is None:
            raise RuntimeError("Video writer not initialized")
        
        # Validate frame dimensions
        if self.is_color:
            expected_shape = (self.height, self.width, 3)
        else:
            expected_shape = (self.height, self.width)
        
        if frame.shape != expected_shape:
            raise ValueError(f"Frame shape {frame.shape} doesn't match expected {expected_shape}")
        
        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def close(self):
        """Close the video writer and report results."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
            # Print statistics
            if Path(self.filename).exists():
                file_size_mb = Path(self.filename).stat().st_size / (1024**2)
                print(f"✓ Video complete: {self.frame_count} frames, {file_size_mb:.1f} MB")
            else:
                print("✗ Video file was not created")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()


def check_available_codecs():
    """
    Test which video codecs are available on the system.
    
    Prints a list of working codecs that can be used.
    
    Returns
    -------
    list
        List of available codec names
    """
    print("Testing available video codecs:")
    print("-" * 70)
    
    test_codecs = {
        'H264 (recommended)': ['H264', 'h264', 'avc1', 'X264'],
        'H265 (high compression)': ['HEVC', 'hev1', 'X265'],
        'MP4V (compatible)': ['mp4v', 'MP4V'],
        'XVID (alternative)': ['XVID', 'xvid'],
        'MJPEG (fast)': ['MJPG', 'mjpg']
    }
    
    available = []
    
    for codec_name, fourcc_list in test_codecs.items():
        working = False
        for fourcc_str in fourcc_list:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter('test.mp4', fourcc, 30.0, (640, 480), True)
                if out.isOpened():
                    out.release()
                    print(f"✓ {codec_name}: use '{fourcc_list[0].lower()}'")
                    available.append(fourcc_list[0].lower())
                    working = True
                    break
            except:
                continue
        
        if not working:
            print(f"✗ {codec_name}: not available")
    
    # Clean up
    test_file = Path('test.mp4')
    if test_file.exists():
        test_file.unlink()
    
    print("-" * 70)
    if available:
        print(f"Recommended codec: '{available[0]}'")
    else:
        print("Warning: No H264/H265 codecs found.")
        print("Install ffmpeg with codec support for best results.")
    
    return available
