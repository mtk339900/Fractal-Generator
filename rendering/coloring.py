"""
Advanced coloring algorithms and palette management for fractal rendering.

This module provides various coloring schemes including escape-time coloring,
smooth/continuous coloring, histogram coloring, and custom palette support.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import get_cmap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available - some coloring features disabled")

from ..core.math_functions import IterationResult

logger = logging.getLogger(__name__)


@dataclass
class ColorRGB:
    """RGB color representation."""
    r: float
    g: float
    b: float
    
    def __post_init__(self):
        """Validate RGB values."""
        for component in [self.r, self.g, self.b]:
            if not 0 <= component <= 1:
                raise ValueError("RGB components must be between 0 and 1")
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)
    
    def to_uint8_tuple(self) -> Tuple[int, int, int]:
        """Convert to 8-bit RGB tuple."""
        return (int(self.r * 255), int(self.g * 255), int(self.b * 255))
    
    def __add__(self, other: 'ColorRGB') -> 'ColorRGB':
        """Add two colors component-wise."""
        return ColorRGB(
            min(1.0, self.r + other.r),
            min(1.0, self.g + other.g),
            min(1.0, self.b + other.b)
        )
    
    def __mul__(self, scalar: float) -> 'ColorRGB':
        """Multiply color by scalar."""
        return ColorRGB(
            min(1.0, self.r * scalar),
            min(1.0, self.g * scalar),
            min(1.0, self.b * scalar)
        )


class Palette:
    """Color palette management and interpolation."""
    
    def __init__(self, colors: List[Union[ColorRGB, Tuple[float, float, float]]], 
                 name: str = "Custom"):
        """
        Initialize color palette.
        
        Args:
            colors: List of colors in the palette
            name: Human-readable name for the palette
        """
        self.name = name
        self.colors = []
        
        for color in colors:
            if isinstance(color, ColorRGB):
                self.colors.append(color)
            elif isinstance(color, (tuple, list)) and len(color) == 3:
                self.colors.append(ColorRGB(*color))
            else:
                raise ValueError(f"Invalid color format: {color}")
        
        if len(self.colors) < 2:
            raise ValueError("Palette must contain at least 2 colors")
    
    def interpolate(self, t: Union[float, np.ndarray]) -> Union[ColorRGB, np.ndarray]:
        """
        Interpolate color at position t (0-1).
        
        Args:
            t: Position in palette (0-1) or array of positions
            
        Returns:
            Interpolated color(s)
        """
        if isinstance(t, (int, float)):
            return self._interpolate_single(t)
        else:
            return self._interpolate_array(t)
    
    def _interpolate_single(self, t: float) -> ColorRGB:
        """Interpolate single color value."""
        t = np.clip(t, 0.0, 1.0)
        
        # Map t to color segments
        segment_size = 1.0 / (len(self.colors) - 1)
        segment_idx = int(t / segment_size)
        
        # Handle edge case
        if segment_idx >= len(self.colors) - 1:
            return self.colors[-1]
        
        # Local interpolation within segment
        local_t = (t - segment_idx * segment_size) / segment_size
        
        color1 = self.colors[segment_idx]
        color2 = self.colors[segment_idx + 1]
        
        return ColorRGB(
            color1.r + local_t * (color2.r - color1.r),
            color1.g + local_t * (color2.g - color1.g),
            color1.b + local_t * (color2.b - color1.b)
        )
    
    def _interpolate_array(self, t: np.ndarray) -> np.ndarray:
        """Interpolate array of color values efficiently."""
        t = np.clip(t, 0.0, 1.0)
        
        # Create RGB output array
        rgb_array = np.zeros((t.shape[0], t.shape[1], 3))
        
        # Convert palette to arrays for vectorized operations
        palette_r = np.array([c.r for c in self.colors])
        palette_g = np.array([c.g for c in self.colors])
        palette_b = np.array([c.b for c in self.colors])
        
        # Map t values to segments
        segment_size = 1.0 / (len(self.colors) - 1)
        segment_indices = np.floor(t / segment_size).astype(int)
        segment_indices = np.clip(segment_indices, 0, len(self.colors) - 2)
        
        # Local interpolation parameter
        local_t = (t - segment_indices * segment_size) / segment_size
        
        # Vectorized interpolation
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                idx = segment_indices[i, j]
                lt = local_t[i, j]
                
                rgb_array[i, j, 0] = palette_r[idx] + lt * (palette_r[idx + 1] - palette_r[idx])
                rgb_array[i, j, 1] = palette_g[idx] + lt * (palette_g[idx + 1] - palette_g[idx])
                rgb_array[i, j, 2] = palette_b[idx] + lt * (palette_b[idx + 1] - palette_b[idx])
        
        return rgb_array
    
    def to_matplotlib_colormap(self, n_colors: int = 256):
        """Convert palette to matplotlib colormap."""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib required for colormap conversion")
        
        # Generate color samples
        t_values = np.linspace(0, 1, n_colors)
        colors = []
        
        for t in t_values:
            color = self._interpolate_single(t)
            colors.append(color.to_tuple())
        
        return mcolors.ListedColormap(colors, name=self.name)
    
    @classmethod
    def from_matplotlib(cls, cmap_name: str, n_samples: int = 32) -> 'Palette':
        """Create palette from matplotlib colormap."""
        if not MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib required for colormap import")
        
        cmap = get_cmap(cmap_name)
        t_values = np.linspace(0, 1, n_samples)
        colors = []
        
        for t in t_values:
            rgba = cmap(t)
            colors.append(ColorRGB(rgba[0], rgba[1], rgba[2]))
        
        return cls(colors, name=f"From_{cmap_name}")
    
    def save_to_file(self, filepath: Path) -> None:
        """Save palette to file in GPL format."""
        with open(filepath, 'w') as f:
            f.write("GIMP Palette\n")
            f.write(f"Name: {self.name}\n")
            f.write("#\n")
            
            for i, color in enumerate(self.colors):
                r, g, b = color.to_uint8_tuple()
                f.write(f"{r:3d} {g:3d} {b:3d} Color_{i}\n")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'Palette':
        """Load palette from GPL file."""
        colors = []
        name = "Loaded_Palette"
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line and not line.startswith("#") and not line.startswith("GIMP"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                            colors.append(ColorRGB(r/255.0, g/255.0, b/255.0))
                        except ValueError:
                            continue
        
        if not colors:
            raise ValueError(f"No valid colors found in {filepath}")
        
        return cls(colors, name)


class ColoringAlgorithm(ABC):
    """Abstract base class for coloring algorithms."""
    
    @abstractmethod
    def apply(self, result: IterationResult, palette: Palette, **kwargs) -> np.ndarray:
        """
        Apply coloring algorithm to iteration result.
        
        Args:
            result: Fractal iteration result
            palette: Color palette to use
            **kwargs: Algorithm-specific parameters
            
        Returns:
            RGB image array (height, width, 3)
        """
        pass


class EscapeTimeColoring(ColoringAlgorithm):
    """Basic escape-time coloring algorithm."""
    
    def apply(self, result: IterationResult, palette: Palette, 
              max_iter: int = 1000, inside_color: Optional[ColorRGB] = None) -> np.ndarray:
        """
        Apply escape-time coloring.
        
        Args:
            result: Iteration result
            palette: Color palette
            max_iter: Maximum iterations for normalization
            inside_color: Color for points that didn't escape
            
        Returns:
            RGB image array
        """
        if inside_color is None:
            inside_color = ColorRGB(0, 0, 0)  # Black
        
        # Normalize iteration counts
        normalized = result.iterations.astype(np.float64) / max_iter
        
        # Apply palette
        rgb_image = palette.interpolate(normalized)
        
        # Set inside color for non-escaped points
        mask = ~result.escaped
        if np.any(mask):
            inside_rgb = inside_color.to_tuple()
            rgb_image[mask, 0] = inside_rgb[0]
            rgb_image[mask, 1] = inside_rgb[1]
            rgb_image[mask, 2] = inside_rgb[2]
        
        return rgb_image


class SmoothColoring(ColoringAlgorithm):
    """Smooth/continuous coloring algorithm for anti-aliased results."""
    
    def apply(self, result: IterationResult, palette: Palette,
              max_iter: int = 1000, inside_color: Optional[ColorRGB] = None) -> np.ndarray:
        """
        Apply smooth coloring using continuous iteration counts.
        
        Args:
            result: Iteration result with final values
            palette: Color palette
            max_iter: Maximum iterations
            inside_color: Color for non-escaped points
            
        Returns:
            RGB image array
        """
        if inside_color is None:
            inside_color = ColorRGB(0, 0, 0)
        
        if result.final_values is None:
            logger.warning("No final values available, falling back to escape-time coloring")
            escape_coloring = EscapeTimeColoring()
            return escape_coloring.apply(result, palette, max_iter, inside_color)
        
        # Calculate smooth iteration counts
        smooth_iterations = result.get_normalized_iterations(max_iter)
        normalized = smooth_iterations / max_iter
        
        # Apply palette
        rgb_image = palette.interpolate(normalized)
        
        # Set inside color for non-escaped points
        mask = ~result.escaped
        if np.any(mask):
            inside_rgb = inside_color.to_tuple()
            rgb_image[mask, 0] = inside_rgb[0]
            rgb_image[mask, 1] = inside_rgb[1]
            rgb_image[mask, 2] = inside_rgb[2]
        
        return rgb_image


class HistogramColoring(ColoringAlgorithm):
    """Histogram equalization coloring for better contrast."""
    
    def apply(self, result: IterationResult, palette: Palette,
              max_iter: int = 1000, inside_color: Optional[ColorRGB] = None) -> np.ndarray:
        """
        Apply histogram-equalized coloring.
        
        Args:
            result: Iteration result
            palette: Color palette
            max_iter: Maximum iterations
            inside_color: Color for non-escaped points
            
        Returns:
            RGB image array
        """
        if inside_color is None:
            inside_color = ColorRGB(0, 0, 0)
        
        # Get escaped points only for histogram calculation
        escaped_iterations = result.iterations[result.escaped]
        
        if len(escaped_iterations) == 0:
            # All points are inside - return solid inside color
            rgb_image = np.full((*result.shape, 3), inside_color.to_tuple())
            return rgb_image
        
        # Calculate histogram
        hist, bins = np.histogram(escaped_iterations, bins=max_iter, range=(0, max_iter))
        
        # Calculate cumulative distribution
        cdf = np.cumsum(hist).astype(np.float64)
        cdf_normalized = cdf / cdf[-1]
        
        # Map iteration values to equalized values
        equalized = np.zeros_like(result.iterations, dtype=np.float64)
        
        for i in range(len(bins) - 1):
            mask = (result.iterations >= bins[i]) & (result.iterations < bins[i + 1]) & result.escaped
            if np.any(mask):
                equalized[mask] = cdf_normalized[i]
        
        # Apply palette
        rgb_image = palette.interpolate(equalized)
        
        # Set inside color
        mask = ~result.escaped
        if np.any(mask):
            inside_rgb = inside_color.to_tuple()
            rgb_image[mask, 0] = inside_rgb[0]
            rgb_image[mask, 1] = inside_rgb[1]
            rgb_image[mask, 2] = inside_rgb[2]
        
        return rgb_image


class OrbitTrapColoring(ColoringAlgorithm):
    """Orbit trap coloring using geometric shapes."""
    
    def __init__(self, trap_function: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize orbit trap coloring.
        
        Args:
            trap_function: Function that calculates distance to trap for complex values
        """
        self.trap_function = trap_function
    
    def apply(self, result: IterationResult, palette: Palette,
              max_iter: int = 1000, inside_color: Optional[ColorRGB] = None) -> np.ndarray:
        """
        Apply orbit trap coloring.
        
        Args:
            result: Iteration result with final values
            palette: Color palette
            max_iter: Maximum iterations
            inside_color: Color for non-escaped points
            
        Returns:
            RGB image array
        """
        if inside_color is None:
            inside_color = ColorRGB(0, 0, 0)
        
        if result.final_values is None:
            logger.warning("No final values available for orbit trap coloring")
            escape_coloring = EscapeTimeColoring()
            return escape_coloring.apply(result, palette, max_iter, inside_color)
        
        # Calculate trap distances
        trap_distances = self.trap_function(result.final_values)
        
        # Normalize distances
        max_distance = np.max(trap_distances[result.escaped])
        if max_distance > 0:
            normalized_distances = trap_distances / max_distance
        else:
            normalized_distances = trap_distances
        
        # Apply palette
        rgb_image = palette.interpolate(normalized_distances)
        
        # Set inside color
        mask = ~result.escaped
        if np.any(mask):
            inside_rgb = inside_color.to_tuple()
            rgb_image[mask, 0] = inside_rgb[0]
            rgb_image[mask, 1] = inside_rgb[1]
            rgb_image[mask, 2] = inside_rgb[2]
        
        return rgb_image


class ColoringEngine:
    """Main engine for applying coloring algorithms."""
    
    def __init__(self):
        """Initialize coloring engine with built-in algorithms."""
        self.algorithms = {
            'escape_time': EscapeTimeColoring(),
            'smooth': SmoothColoring(),
            'histogram': HistogramColoring(),
        }
        
        # Built-in palettes
        self.palettes = self._create_builtin_palettes()
    
    def _create_builtin_palettes(self) -> Dict[str, Palette]:
        """Create built-in color palettes."""
        palettes = {}
        
        # Classic hot palette
        palettes['hot'] = Palette([
            ColorRGB(0, 0, 0),      # Black
            ColorRGB(1, 0, 0),      # Red
            ColorRGB(1, 1, 0),      # Yellow
            ColorRGB(1, 1, 1),      # White
        ], name="Hot")
        
        # Cool palette
        palettes['cool'] = Palette([
            ColorRGB(0, 0, 0),      # Black
            ColorRGB(0, 0, 1),      # Blue
            ColorRGB(0, 1, 1),      # Cyan
            ColorRGB(1, 1, 1),      # White
        ], name="Cool")
        
        # Grayscale
        palettes['gray'] = Palette([
            ColorRGB(0, 0, 0),      # Black
            ColorRGB(1, 1, 1),      # White
        ], name="Grayscale")
        
        # Fire
        palettes['fire'] = Palette([
            ColorRGB(0, 0, 0),          # Black
            ColorRGB(0.5, 0, 0),        # Dark red
            ColorRGB(1, 0, 0),          # Red
            ColorRGB(1, 0.5, 0),        # Orange
            ColorRGB(1, 1, 0),          # Yellow
            ColorRGB(1, 1, 1),          # White
        ], name="Fire")
        
        # Ocean
        palettes['ocean'] = Palette([
            ColorRGB(0, 0, 0.2),        # Deep blue
            ColorRGB(0, 0, 0.8),        # Blue
            ColorRGB(0, 0.5, 1),        # Light blue
            ColorRGB(0, 1, 1),          # Cyan
            ColorRGB(0.5, 1, 1),        # Light cyan
            ColorRGB(1, 1, 1),          # White
        ], name="Ocean")
        
        # Rainbow
        palettes['rainbow'] = Palette([
            ColorRGB(1, 0, 0),      # Red
            ColorRGB(1, 0.5, 0),    # Orange
            ColorRGB(1, 1, 0),      # Yellow
            ColorRGB(0, 1, 0),      # Green
            ColorRGB(0, 1, 1),      # Cyan
            ColorRGB(0, 0, 1),      # Blue
            ColorRGB(0.5, 0, 1),    # Purple
        ], name="Rainbow")
        
        # Try to add matplotlib-based palettes if available
        if MATPLOTLIB_AVAILABLE:
            try:
                matplotlib_palettes = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
                for name in matplotlib_palettes:
                    try:
                        palettes[name] = Palette.from_matplotlib(name, 32)
                    except Exception as e:
                        logger.warning(f"Could not load matplotlib palette {name}: {e}")
            except Exception as e:
                logger.warning(f"Error loading matplotlib palettes: {e}")
        
        return palettes
    
    def add_algorithm(self, name: str, algorithm: ColoringAlgorithm) -> None:
        """Add a custom coloring algorithm."""
        self.algorithms[name] = algorithm
        logger.info(f"Added coloring algorithm: {name}")
    
    def add_palette(self, name: str, palette: Palette) -> None:
        """Add a custom color palette."""
        self.palettes[name] = palette
        logger.info(f"Added color palette: {name}")
    
    def get_algorithm(self, name: str) -> ColoringAlgorithm:
        """Get coloring algorithm by name."""
        if name not in self.algorithms:
            available = ', '.join(self.algorithms.keys())
            raise ValueError(f"Unknown coloring algorithm '{name}'. Available: {available}")
        return self.algorithms[name]
    
    def get_palette(self, name: str) -> Palette:
        """Get color palette by name."""
        if name not in self.palettes:
            available = ', '.join(self.palettes.keys())
            raise ValueError(f"Unknown color palette '{name}'. Available: {available}")
        return self.palettes[name]
    
    def render_color_image(self, result: IterationResult, algorithm: str = 'smooth',
                          palette: str = 'hot', max_iter: int = 1000,
                          inside_color: Optional[ColorRGB] = None, **kwargs) -> np.ndarray:
        """
        Render colored image from iteration result.
        
        Args:
            result: Fractal iteration result
            algorithm: Coloring algorithm name
            palette: Color palette name
            max_iter: Maximum iterations for normalization
            inside_color: Color for non-escaped points
            **kwargs: Additional algorithm parameters
            
        Returns:
            RGB image array (height, width, 3) with values 0-1
        """
        coloring_alg = self.get_algorithm(algorithm)
        color_palette = self.get_palette(palette)
        
        return coloring_alg.apply(result, color_palette, max_iter, inside_color, **kwargs)
    
    def list_algorithms(self) -> List[str]:
        """Get list of available coloring algorithms."""
        return list(self.algorithms.keys())
    
    def list_palettes(self) -> List[str]:
        """Get list of available color palettes."""
        return list(self.palettes.keys())


# Predefined trap functions for orbit trap coloring
def point_trap(center: complex = 0+0j):
    """Create point trap function."""
    def trap_func(z: np.ndarray) -> np.ndarray:
        return np.abs(z - center)
    return trap_func


def line_trap(slope: float = 1.0, intercept: float = 0.0):
    """Create line trap function."""
    def trap_func(z: np.ndarray) -> np.ndarray:
        # Distance from point to line ax + by + c = 0
        # Line equation: y = slope * x + intercept -> slope*x - y + intercept = 0
        a, b, c = slope, -1, intercept
        distance = np.abs(a * z.real + b * z.imag + c) / np.sqrt(a*a + b*b)
        return distance
    return trap_func


def circle_trap(center: complex = 0+0j, radius: float = 0.5):
    """Create circle trap function."""
    def trap_func(z: np.ndarray) -> np.ndarray:
        return np.abs(np.abs(z - center) - radius)
    return trap_func


def cross_trap(center: complex = 0+0j, thickness: float = 0.1):
    """Create cross-shaped trap function."""
    def trap_func(z: np.ndarray) -> np.ndarray:
        rel_z = z - center
        horizontal_dist = np.abs(rel_z.imag)
        vertical_dist = np.abs(rel_z.real)
        return np.minimum(horizontal_dist, vertical_dist)
    return trap_func
