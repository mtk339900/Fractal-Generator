"""
Arbitrary precision mathematics for deep fractal zooms.

This module provides high-precision arithmetic capabilities using mpmath
for extreme zoom levels where standard floating-point precision is insufficient.
"""

import numpy as np
from typing import Union, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Optional import of mpmath for arbitrary precision
try:
    import mpmath as mp
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False
    logger.warning("mpmath not available - arbitrary precision disabled")


class PrecisionConfig:
    """Configuration for precision levels and arithmetic operations."""
    
    def __init__(self, precision: Union[str, int] = 'double'):
        """
        Initialize precision configuration.
        
        Args:
            precision: Either 'single', 'double', 'quad', or number of decimal places
        """
        self.precision = precision
        self._setup_precision()
    
    def _setup_precision(self):
        """Setup precision parameters based on configuration."""
        if isinstance(self.precision, str):
            if self.precision == 'single':
                self.dtype = np.complex64
                self.real_dtype = np.float32
                self.use_mpmath = False
                self.decimal_places = 7
            elif self.precision == 'double':
                self.dtype = np.complex128
                self.real_dtype = np.float64
                self.use_mpmath = False
                self.decimal_places = 15
            elif self.precision == 'quad' and MPMATH_AVAILABLE:
                self.dtype = None  # Use mpmath
                self.real_dtype = None
                self.use_mpmath = True
                self.decimal_places = 34
                mp.mp.dps = 34
            elif self.precision == 'quad':
                logger.warning("Quad precision requested but mpmath not available, using double")
                self.dtype = np.complex128
                self.real_dtype = np.float64
                self.use_mpmath = False
                self.decimal_places = 15
            else:
                raise ValueError(f"Unknown precision type: {self.precision}")
        
        elif isinstance(self.precision, int):
            if not MPMATH_AVAILABLE:
                logger.warning("High precision requested but mpmath not available, using double")
                self.dtype = np.complex128
                self.real_dtype = np.float64
                self.use_mpmath = False
                self.decimal_places = 15
            else:
                self.dtype = None
                self.real_dtype = None
                self.use_mpmath = True
                self.decimal_places = max(15, self.precision)
                mp.mp.dps = self.decimal_places
        else:
            raise ValueError(f"Invalid precision specification: {self.precision}")
    
    def format_number(self, value: Union[float, complex]) -> str:
        """Format a number according to the precision configuration."""
        if self.use_mpmath and MPMATH_AVAILABLE:
            if isinstance(value, complex):
                real_str = mp.nstr(mp.mpf(value.real), n=min(10, self.decimal_places))
                imag_str = mp.nstr(mp.mpf(value.imag), n=min(10, self.decimal_places))
                return f"{real_str} + {imag_str}i"
            else:
                return mp.nstr(mp.mpf(value), n=min(10, self.decimal_places))
        else:
            precision = min(8, self.decimal_places)
            if isinstance(value, complex):
                return f"{value.real:.{precision}g} + {value.imag:.{precision}g}i"
            else:
                return f"{value:.{precision}g}"


class HighPrecisionComplex:
    """High-precision complex number implementation using mpmath."""
    
    def __init__(self, real: Union[str, float, 'mp.mpf'], 
                 imag: Union[str, float, 'mp.mpf'] = 0):
        """
        Initialize high-precision complex number.
        
        Args:
            real: Real part
            imag: Imaginary part
        """
        if not MPMATH_AVAILABLE:
            raise RuntimeError("mpmath required for high-precision arithmetic")
        
        self.real = mp.mpf(real)
        self.imag = mp.mpf(imag)
    
    def __add__(self, other: 'HighPrecisionComplex') -> 'HighPrecisionComplex':
        """Add two high-precision complex numbers."""
        return HighPrecisionComplex(
            self.real + other.real,
            self.imag + other.imag
        )
    
    def __sub__(self, other: 'HighPrecisionComplex') -> 'HighPrecisionComplex':
        """Subtract two high-precision complex numbers."""
        return HighPrecisionComplex(
            self.real - other.real,
            self.imag - other.imag
        )
    
    def __mul__(self, other: 'HighPrecisionComplex') -> 'HighPrecisionComplex':
        """Multiply two high-precision complex numbers."""
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return HighPrecisionComplex(real, imag)
    
    def __pow__(self, power: Union[int, float]) -> 'HighPrecisionComplex':
        """Raise complex number to a power."""
        if power == 2:
            # Optimized squaring
            real = self.real * self.real - self.imag * self.imag
            imag = 2 * self.real * self.imag
            return HighPrecisionComplex(real, imag)
        else:
            # General power using polar form
            magnitude = self.abs()
            angle = self.arg()
            new_magnitude = magnitude ** power
            new_angle = angle * power
            
            real = new_magnitude * mp.cos(new_angle)
            imag = new_magnitude * mp.sin(new_angle)
            return HighPrecisionComplex(real, imag)
    
    def abs(self) -> 'mp.mpf':
        """Calculate absolute value (magnitude)."""
        return mp.sqrt(self.real * self.real + self.imag * self.imag)
    
    def abs_squared(self) -> 'mp.mpf':
        """Calculate squared absolute value for efficiency."""
        return self.real * self.real + self.imag * self.imag
    
    def arg(self) -> 'mp.mpf':
        """Calculate argument (angle) of complex number."""
        return mp.atan2(self.imag, self.real)
    
    def conjugate(self) -> 'HighPrecisionComplex':
        """Return complex conjugate."""
        return HighPrecisionComplex(self.real, -self.imag)
    
    def to_tuple(self) -> Tuple['mp.mpf', 'mp.mpf']:
        """Convert to tuple of (real, imag)."""
        return (self.real, self.imag)
    
    def to_complex(self) -> complex:
        """Convert to standard Python complex (may lose precision)."""
        return complex(float(self.real), float(self.imag))
    
    def __str__(self) -> str:
        """String representation."""
        real_str = mp.nstr(self.real, n=10)
        imag_str = mp.nstr(self.imag, n=10)
        if self.imag >= 0:
            return f"{real_str} + {imag_str}i"
        else:
            return f"{real_str} - {mp.nstr(-self.imag, n=10)}i"
    
    def __repr__(self) -> str:
        return f"HighPrecisionComplex({self.real}, {self.imag})"


class ArbitraryPrecisionPlane:
    """High-precision version of ComplexPlane for deep zooms."""
    
    def __init__(self, xmin: Union[str, float], xmax: Union[str, float],
                 ymin: Union[str, float], ymax: Union[str, float],
                 width: int, height: int):
        """
        Initialize high-precision complex plane.
        
        Args:
            xmin, xmax, ymin, ymax: Plane bounds (can be strings for exact precision)
            width, height: Resolution in pixels
        """
        if not MPMATH_AVAILABLE:
            raise RuntimeError("mpmath required for arbitrary precision")
        
        self.xmin = mp.mpf(xmin)
        self.xmax = mp.mpf(xmax)
        self.ymin = mp.mpf(ymin)
        self.ymax = mp.mpf(ymax)
        self.width = width
        self.height = height
        
        # Calculate scaling factors with high precision
        self.x_scale = (self.xmax - self.xmin) / width
        self.y_scale = (self.ymax - self.ymin) / height
        
        # Validate bounds
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Invalid bounds")
    
    def pixel_to_complex(self, px: int, py: int) -> HighPrecisionComplex:
        """Convert pixel coordinates to high-precision complex number."""
        real = self.xmin + px * self.x_scale
        imag = self.ymin + py * self.y_scale
        return HighPrecisionComplex(real, imag)
    
    def create_coordinate_grid(self) -> Tuple[list, list]:
        """Create coordinate grids for the plane."""
        x_coords = []
        y_coords = []
        
        for px in range(self.width):
            x = self.xmin + px * self.x_scale
            x_coords.append(x)
        
        for py in range(self.height):
            y = self.ymin + py * self.y_scale
            y_coords.append(y)
        
        return x_coords, y_coords


class HighPrecisionIterator:
    """High-precision fractal iterator for deep zooms."""
    
    def __init__(self, max_iter: int = 1000, escape_radius: Union[float, str] = 2.0):
        """
        Initialize high-precision iterator.
        
        Args:
            max_iter: Maximum iterations
            escape_radius: Escape radius (can be string for exact precision)
        """
        if not MPMATH_AVAILABLE:
            raise RuntimeError("mpmath required for high-precision iteration")
        
        self.max_iter = max_iter
        self.escape_radius = mp.mpf(escape_radius)
        self.escape_radius_sq = self.escape_radius ** 2
    
    def mandelbrot_point(self, c: HighPrecisionComplex, 
                        z0: Optional[HighPrecisionComplex] = None) -> Tuple[int, HighPrecisionComplex]:
        """
        Compute Mandelbrot iteration for a single point.
        
        Args:
            c: Complex parameter
            z0: Initial z value
            
        Returns:
            Tuple of (iterations, final_z)
        """
        if z0 is None:
            z = HighPrecisionComplex(0, 0)
        else:
            z = z0
        
        for i in range(self.max_iter):
            if z.abs_squared() > self.escape_radius_sq:
                return i, z
            z = z * z + c
        
        return self.max_iter, z
    
    def julia_point(self, z: HighPrecisionComplex, 
                   c: HighPrecisionComplex) -> Tuple[int, HighPrecisionComplex]:
        """
        Compute Julia iteration for a single point.
        
        Args:
            z: Initial complex value
            c: Julia constant
            
        Returns:
            Tuple of (iterations, final_z)
        """
        for i in range(self.max_iter):
            if z.abs_squared() > self.escape_radius_sq:
                return i, z
            z = z * z + c
        
        return self.max_iter, z
    
    def render_mandelbrot_region(self, plane: ArbitraryPrecisionPlane,
                                z0: Optional[HighPrecisionComplex] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render Mandelbrot set for a region with high precision.
        
        Args:
            plane: High-precision plane definition
            z0: Initial z value
            
        Returns:
            Tuple of (iterations_array, convergence_array)
        """
        iterations = np.zeros((plane.height, plane.width), dtype=np.int32)
        converged = np.zeros((plane.height, plane.width), dtype=bool)
        
        logger.info(f"Starting high-precision Mandelbrot render: {plane.width}x{plane.height}")
        
        for py in range(plane.height):
            if py % 100 == 0:
                logger.info(f"Processing row {py}/{plane.height}")
            
            for px in range(plane.width):
                c = plane.pixel_to_complex(px, py)
                iter_count, final_z = self.mandelbrot_point(c, z0)
                
                iterations[py, px] = iter_count
                converged[py, px] = (iter_count < self.max_iter)
        
        logger.info("High-precision Mandelbrot render complete")
        return iterations, converged
    
    def render_julia_region(self, plane: ArbitraryPrecisionPlane,
                           c: HighPrecisionComplex) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render Julia set for a region with high precision.
        
        Args:
            plane: High-precision plane definition
            c: Julia constant
            
        Returns:
            Tuple of (iterations_array, convergence_array)
        """
        iterations = np.zeros((plane.height, plane.width), dtype=np.int32)
        converged = np.zeros((plane.height, plane.width), dtype=bool)
        
        logger.info(f"Starting high-precision Julia render: {plane.width}x{plane.height}")
        
        for py in range(plane.height):
            if py % 100 == 0:
                logger.info(f"Processing row {py}/{plane.height}")
            
            for px in range(plane.width):
                z = plane.pixel_to_complex(px, py)
                iter_count, final_z = self.julia_point(z, c)
                
                iterations[py, px] = iter_count
                converged[py, px] = (iter_count < self.max_iter)
        
        logger.info("High-precision Julia render complete")
        return iterations, converged


def detect_precision_need(zoom_level: float, center_real: float, center_imag: float) -> str:
    """
    Detect if high precision is needed based on zoom level and position.
    
    Args:
        zoom_level: Current zoom level (higher = more zoomed in)
        center_real: Real part of zoom center
        center_imag: Imaginary part of zoom center
        
    Returns:
        Recommended precision level ('single', 'double', 'quad', or decimal places)
    """
    # Calculate effective precision needed
    digits_needed = max(0, np.log10(zoom_level) + 2)
    
    if digits_needed <= 6:
        return 'single'
    elif digits_needed <= 14:
        return 'double'
    elif digits_needed <= 32:
        return 'quad'
    else:
        # Need arbitrary precision
        return int(digits_needed + 10)


def convert_bounds_to_high_precision(xmin: float, xmax: float, ymin: float, ymax: float,
                                   zoom_center: Tuple[float, float],
                                   zoom_factor: float) -> Tuple[str, str, str, str]:
    """
    Convert standard bounds to high-precision string representation.
    
    Args:
        xmin, xmax, ymin, ymax: Original bounds
        zoom_center: Center point for zoom
        zoom_factor: Zoom multiplication factor
        
    Returns:
        Tuple of high-precision bound strings
    """
    cx, cy = zoom_center
    
    # Calculate new bounds with high precision
    width = (xmax - xmin) / zoom_factor
    height = (ymax - ymin) / zoom_factor
    
    new_xmin = cx - width / 2
    new_xmax = cx + width / 2
    new_ymin = cy - height / 2
    new_ymax = cy + height / 2
    
    # Convert to high-precision strings
    return (
        f"{new_xmin:.50g}",
        f"{new_xmax:.50g}",
        f"{new_ymin:.50g}",
        f"{new_ymax:.50g}"
    )
