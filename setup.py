"""
FastVideo Setup Script
====================

Installation script for the FastVideo package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="fastvideo",
    version="2.0.0",
    author="Carlo",
    author_email="",
    description="Scientific microscopy video creation with velocity field overlays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Multimedia :: Video :: Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0",
        "opencv-python>=4.0.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "gpu": ["cupy>=8.0.0"],
        "full": [
            "cupy>=8.0.0",
            "matplotlib>=3.0.0",
            "scipy>=1.5.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fastvideo-check=fastvideo:check_available_codecs",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
