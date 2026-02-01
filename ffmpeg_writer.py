"""
FFmpeg Video Writer
===================

Robust FFmpeg-based video writer with error handling and diagnostics.

Author: Carlo
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def check_ffmpeg_codecs() -> dict:
    """
    Check which video codecs are available in FFmpeg installation.
    
    Returns
    -------
    dict
        Dictionary with codec names as keys and availability as boolean values
        Example: {'h264': True, 'h265': True, 'h264_nvenc': False}
    
    Examples
    --------
    >>> codecs = check_ffmpeg_codecs()
    >>> if codecs['h265_nvenc']:
    ...     print("GPU encoding available!")
    >>> else:
    ...     print("Using CPU encoding")
    
    Notes
    -----
    NVENC codecs require NVIDIA GPU with hardware encoding support.
    Check with: nvidia-smi
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-codecs'],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout
        
        codecs = {
            'h264': 'libx264' in output,
            'h264_nvenc': 'h264_nvenc' in output,
            'h265': 'libx265' in output or 'hevc' in output,
            'h265_nvenc': 'hevc_nvenc' in output,
        }
        
        return codecs
    
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return {'h264': False, 'h265': False, 'h264_nvenc': False, 'h265_nvenc': False}
    
    except Exception as e:
        print(f"Warning: Could not check FFmpeg codecs: {e}")
        return {'h264': True, 'h265': False, 'h264_nvenc': False, 'h265_nvenc': False}


def get_nvenc_preset(preset: str) -> str:
    """
    Convert preset to NVENC-compatible format.
    
    NVENC presets changed between FFmpeg versions:
    - Older versions: p1-p7
    - Newer versions: fast, medium, slow, etc.
    
    Parameters
    ----------
    preset : str
        User-specified preset
    
    Returns
    -------
    str
        NVENC-compatible preset name
    """
    # Map common presets to NVENC equivalents
    nvenc_preset_map = {
        'ultrafast': 'fast',
        'faster': 'fast',
        'fast': 'fast',
        'medium': 'medium',
        'slow': 'slow',
        'slower': 'slow',
        'veryslow': 'slow',
        # Also accept p-style presets directly
        'p1': 'fast',
        'p2': 'fast',
        'p3': 'fast',
        'p4': 'medium',
        'p5': 'medium',
        'p6': 'slow',
        'p7': 'slow',
    }
    
    return nvenc_preset_map.get(preset.lower(), 'medium')


def validate_ffmpeg_params(
    width: int,
    height: int,
    fps: float,
    codec: str
) -> bool:
    """
    Validate FFmpeg parameters before starting video creation.
    
    Parameters
    ----------
    width, height : int
        Video dimensions in pixels
    fps : float
        Frames per second
    codec : str
        Video codec name
    
    Returns
    -------
    bool
        True if parameters are valid, False otherwise
    
    Notes
    -----
    Checks performed:
    - Dimensions must be even (required for YUV420p)
    - FPS must be reasonable (1-120)
    - Codec must be available in FFmpeg
    """
    valid = True
    
    # Check dimensions
    if width % 2 != 0 or height % 2 != 0:
        print(f"Error: Width and height must be even numbers")
        print(f"  Current: {width}×{height}")
        print(f"  Suggested: {width - width%2}×{height - height%2}")
        valid = False
    
    # Check FPS
    if fps <= 0 or fps > 120:
        print(f"Warning: Unusual FPS value: {fps}")
        if fps <= 0:
            valid = False
    
    # Check codec availability
    available_codecs = check_ffmpeg_codecs()
    
    codec_map = {
        'h264': 'h264',
        'h265': 'h265',
        'hevc': 'h265',
        'h264_nvenc': 'h264_nvenc',
        'h265_nvenc': 'h265_nvenc',
        'hevc_nvenc': 'h265_nvenc'
    }
    
    codec_key = codec_map.get(codec, codec)
    
    if codec_key not in available_codecs or not available_codecs[codec_key]:
        print(f"Warning: Codec '{codec}' may not be available")
        print(f"Available codecs: {[k for k, v in available_codecs.items() if v]}")
    
    return valid


class RobustFFmpegWriter:
    """
    FFmpeg video writer with comprehensive error handling.
    
    Provides robust video encoding with:
    - Parameter validation before starting
    - FFmpeg stderr capture for debugging
    - Automatic dimension adjustment
    - Buffer management to prevent blocking
    - Proper cleanup on errors
    
    Parameters
    ----------
    filename : str
        Output video file path
    width : int
        Frame width in pixels (will be adjusted to even if needed)
    height : int
        Frame height in pixels (will be adjusted to even if needed)
    fps : float
        Frames per second (1-120)
    bitrate : str, optional
        Target bitrate (e.g., '500k', '2M'). Default: '500k'
    codec : str, optional
        Video codec: 'h264', 'h265', 'h264_nvenc', 'h265_nvenc'
        Default: 'h265' (best compression)
    preset : str, optional
        Encoding preset:
        - CPU: 'ultrafast', 'fast', 'medium', 'slow', 'veryslow'
        - NVENC: 'fast', 'medium', 'slow' (automatically converted)
        Default: 'medium'
    pix_fmt : str, optional
        Pixel format. Default: 'yuv420p' (universal compatibility)
    verbose : bool, optional
        Print FFmpeg command and detailed output. Default: False
    
    Attributes
    ----------
    process : subprocess.Popen
        FFmpeg subprocess
    width, height : int
        Actual video dimensions (adjusted to even if needed)
    
    Examples
    --------
    >>> # Basic usage
    >>> with RobustFFmpegWriter('output.mp4', 1024, 1024, 25) as writer:
    ...     for frame in frames:
    ...         writer.write(frame)
    
    >>> # GPU encoding (NVENC)
    >>> writer = RobustFFmpegWriter(
    ...     'output.mp4', 1024, 1024, 25,
    ...     codec='h265_nvenc',
    ...     preset='medium',  # Automatically converted to NVENC preset
    ...     bitrate='2M'
    ... )
    >>> for frame in frames:
    ...     writer.write(frame)
    >>> writer.close()
    
    >>> # High compression
    >>> writer = RobustFFmpegWriter(
    ...     'output.mp4', 1024, 1024, 25,
    ...     codec='h265',
    ...     preset='slow',  # Better compression
    ...     bitrate='300k'  # Lower bitrate
    ... )
    
    Notes
    -----
    - Always use as context manager or call close() to finalize video
    - Frames must be uint8 RGB arrays with shape (height, width, 3)
    - For grayscale, pass 2D array - it will be converted to RGB
    - BrokenPipeError usually means invalid codec settings or FFmpeg crash
    """
    
    def __init__(
        self,
        filename: str,
        width: int,
        height: int,
        fps: float,
        bitrate: str = '500k',
        codec: str = 'h265',
        preset: str = 'medium',
        pix_fmt: str = 'yuv420p',
        verbose: bool = False
    ):
        self.filename = str(filename)
        self.verbose = verbose
        
        # Ensure dimensions are even (required for YUV420p)
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        
        if self.width != width or self.height != height:
            print(f"Adjusted resolution: {width}×{height} → {self.width}×{self.height}")
        
        self.fps = max(1, min(int(fps), 120))
        self.bitrate = bitrate
        self.codec = codec
        self.preset = preset
        self.pix_fmt = pix_fmt
        
        # Validate parameters
        validate_ffmpeg_params(self.width, self.height, self.fps, codec)
        
        # Build command
        self.cmd = self._build_command()
        
        # Start process
        self.process = None
        self._start_process()
    
    def _build_command(self):
        """Build FFmpeg command with codec-specific optimizations."""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',  # Input format
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
        ]
        
        # Codec-specific settings
        if self.codec in ['h265', 'hevc']:
            cmd.extend([
                '-vcodec', 'libx265',
                '-preset', self.preset,
                '-b:v', self.bitrate,
                '-maxrate', self.bitrate,
                '-bufsize', f'{int(self.bitrate[:-1]) * 2}k',
                '-pix_fmt', self.pix_fmt,
                '-x265-params', 'log-level=error',
            ])
        
        elif self.codec in ['h265_nvenc', 'hevc_nvenc']:
            # Convert preset to NVENC-compatible format
            nvenc_preset = get_nvenc_preset(self.preset)
            cmd.extend([
                '-vcodec', 'hevc_nvenc',
                '-preset', nvenc_preset,
                '-b:v', self.bitrate,
                '-maxrate', self.bitrate,
                '-bufsize', f'{int(self.bitrate[:-1]) * 2}k',
                '-pix_fmt', self.pix_fmt,
            ])
            if self.verbose:
                print(f"NVENC: Converting preset '{self.preset}' → '{nvenc_preset}'")
        
        elif self.codec == 'h264_nvenc':
            # Convert preset to NVENC-compatible format
            nvenc_preset = get_nvenc_preset(self.preset)
            cmd.extend([
                '-vcodec', 'h264_nvenc',
                '-preset', nvenc_preset,
                '-b:v', self.bitrate,
                '-maxrate', self.bitrate,
                '-bufsize', f'{int(self.bitrate[:-1]) * 2}k',
                '-pix_fmt', self.pix_fmt,
            ])
            if self.verbose:
                print(f"NVENC: Converting preset '{self.preset}' → '{nvenc_preset}'")
        
        else:  # h264 or fallback
            cmd.extend([
                '-vcodec', 'libx264',
                '-preset', self.preset,
                '-b:v', self.bitrate,
                '-maxrate', self.bitrate,
                '-bufsize', f'{int(self.bitrate[:-1]) * 2}k',
                '-pix_fmt', self.pix_fmt,
            ])
        
        cmd.append(self.filename)
        
        if self.verbose:
            print(f"FFmpeg command: {' '.join(cmd)}")
        
        return cmd
    
    def _start_process(self):
        """Start FFmpeg subprocess."""
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                bufsize=10**8  # Large buffer (100MB)
            )
            
            print(f"FFmpeg writer initialized:")
            print(f"  Output: {self.filename}")
            print(f"  Resolution: {self.width}×{self.height}")
            print(f"  FPS: {self.fps}")
            print(f"  Codec: {self.codec}")
            print(f"  Bitrate: {self.bitrate}")
            
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install:\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg: {e}")
    
    def write(self, frame: np.ndarray):
        """
        Write a single frame to video.
        
        Parameters
        ----------
        frame : ndarray
            Frame to write. Must be:
            - Shape: (height, width, 3) for RGB or (height, width) for grayscale
            - Dtype: uint8 (0-255 range)
            - Same dimensions as specified in __init__
        
        Raises
        ------
        BrokenPipeError
            If FFmpeg process has terminated unexpectedly
        ValueError
            If frame dimensions don't match expected size
        """
        if self.process is None or self.process.poll() is not None:
            self._capture_error()
            raise BrokenPipeError(
                "FFmpeg process terminated unexpectedly. "
                "Check error messages above or try verbose=True"
            )
        
        # Validate frame
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(
                f"Frame dimensions {frame.shape[1]}×{frame.shape[0]} "
                f"don't match expected {self.width}×{self.height}"
            )
        
        # Convert to uint8 RGB
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        if frame.ndim == 2:
            # Grayscale → RGB
            frame = np.stack([frame] * 3, axis=-1)
        
        # Write to FFmpeg stdin
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            self._capture_error()
            raise BrokenPipeError(
                "FFmpeg closed pipe unexpectedly. "
                "Common causes:\n"
                "  1. Invalid codec (try codec='h264')\n"
                "  2. Odd dimensions (ensure even width/height)\n"
                "  3. Missing codec (check with check_ffmpeg_codecs())\n"
                "Run with verbose=True for detailed FFmpeg output."
            )
    
    def _capture_error(self):
        """Capture and display FFmpeg stderr for debugging."""
        if self.process and self.process.stderr:
            try:
                stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                if stderr:
                    print("\n" + "="*70)
                    print("FFmpeg Error Output:")
                    print("="*70)
                    print(stderr)
                    print("="*70 + "\n")
            except:
                pass
    
    def close(self):
        """
        Close writer and finalize video file.
        
        Must be called to properly finish the video encoding.
        """
        if self.process is None:
            return
        
        try:
            # Close stdin to signal end of input
            if self.process.stdin:
                self.process.stdin.close()
            
            # Wait for FFmpeg to finish (max 10 seconds)
            self.process.wait(timeout=10)
            
            # Check exit code
            if self.process.returncode != 0:
                self._capture_error()
                print(f"Warning: FFmpeg exited with code {self.process.returncode}")
            
            # Report file size
            if Path(self.filename).exists():
                file_size = Path(self.filename).stat().st_size / (1024**2)
                print(f"\n✓ Video complete: {file_size:.1f} MB")
            else:
                print(f"\n✗ Video file not created: {self.filename}")
        
        except subprocess.TimeoutExpired:
            print("Warning: FFmpeg did not terminate, killing process")
            self.process.kill()
            self._capture_error()
        
        except Exception as e:
            print(f"Error closing FFmpeg: {e}")
            self._capture_error()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always close properly."""
        self.close()