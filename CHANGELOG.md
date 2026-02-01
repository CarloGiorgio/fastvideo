# Changelog

All notable changes to FastVideo will be documented in this file.

## [2.0.0] - 2025-02-01

### Added
- Complete rewrite and modularization of video creation code
- Comprehensive `VideoWriter` class with automatic codec fallback
- High-level convenience functions: `create_video_simple()`, `create_velocity_video()`
- Automatic arrow scale calibration with `calculate_auto_arrow_scale()`
- Frame range validation and FPS calculation utilities
- Text overlay support with customizable styling
- Quality presets (high/medium/low) for easy configuration
- Extensive documentation and examples
- Support for both grayscale and RGB videos
- GPU-accelerated processing (CuPy optional)
- Backward compatibility with existing code

### Changed
- Unified API across all video creation functions
- Improved error handling with graceful fallbacks
- Better parameter validation
- Cleaner separation between core and advanced features

### Fixed
- Codec availability detection
- Frame dimension validation
- Memory management for large stacks

## [1.0.0] - Previous Version

- Initial implementation with basic video creation
- PIV visualization support
- FFmpeg integration
- Batch processing capabilities

---

[2.0.0]: https://github.com/your-repo/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/your-repo/releases/tag/v1.0.0
