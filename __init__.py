"""
High-quality fractal generation library.

This library provides comprehensive tools for generating Mandelbrot sets, Julia sets,
and other fractals with advanced rendering, coloring, and acceleration features.

Key Features:
- Multiple precision levels (float32, float64, arbitrary precision)
- Advanced coloring algorithms and custom palettes
- GPU acceleration with graceful CPU fallback
- Tile-based rendering for memory efficiency
- Animation and interactive visualization support
- Extensible plugin architecture for custom fractals

Example usage:
    >>> from fractal_generator import FractalRenderer, MandelbrotSet
    >>> fractal = MandelbrotSet()
    >>> renderer = FractalRenderer(width=1920, height=1080)
    >>> image = renderer.render(fractal, bounds=(-2, 1, -1.5, 1.5))
"""

__version__ = "1.0.0"
__author__ = "Fractal Generator Team"

from fractal_generator.core.fractal_types import MandelbrotSet, JuliaSet, BurningShip, Multibrot
from fractal_generator.core.math_functions import FractalIterator
from fractal_generator.rendering.coloring import ColoringEngine, Palette
from fractal_generator.rendering.image_output import ImageExporter
from fractal_generator.tools.tiling import TileRenderer
from fractal_generator.tools.animation import AnimationSequencer
from fractal_generator.io.config import ConfigManager

# Main API classes
from fractal_generator.api import FractalRenderer, RenderConfig

__all__ = [
    "FractalRenderer",
    "RenderConfig",
    "MandelbrotSet",
    "JuliaSet",
    "BurningShip",
    "Multibrot",
    "FractalIterator",
    "ColoringEngine",
    "Palette",
    "ImageExporter",
    "TileRenderer",
    "AnimationSequencer",
    "ConfigManager",
]
