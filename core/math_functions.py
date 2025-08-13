"""
Core mathematical functions for fractal iteration.

This module provides the fundamental iteration algorithms and mathematical
operations required for fractal generation, supporting multiple precision
levels and optimization strategies.
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ComplexPlane:
    """Represents a complex plane region with coordinate mapping utilities."""
    
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float,
                 width: int, height: int):
        """
        Initialize complex plane bounds and resolution.
        
        Args:
            xmin, xmax: Real axis bounds
            ymin, ymax: Imaginary axis bounds
            width, height: Image resolution in pixels
        """
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = width
        self.height = height
        
        # Calculate scaling factors
        self.x_scale = (xmax - xmin) / width
        self.y_scale = (ymax - ymin) / height
        
        # Validate bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounds: min values must be less than max values")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
    
    def create_coordinate_arrays(self, dtype: np.dtype = np.complex128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create coordinate arrays for the complex plane.
        
        Args:
            dtype: Data type for the arrays
            
        Returns:
            Tuple of (real_coords, imag_coords) arrays
        """
        x = np.linspace(self.xmin, self.xmax, self.width, dtype=dtype.type(0).real.dtype)
        y = np.linspace(self.ymin, self.ymax, self.height, dtype=dtype.type(0).real.dtype)
        return np.meshgrid(x, y)
    
    def create_complex_array(self, dtype: np.dtype = np.complex128) -> np.ndarray:
        """
        Create a complex coordinate array for the entire plane.
        
        Args:
            dtype: Complex data type
            
        Returns:
            2D array of complex coordinates
        """
        x, y = self.create_coordinate_arrays(dtype)
        return x + 1j * y
    
    def pixel_to_complex(self, px: int, py: int) -> complex:
        """Convert pixel coordinates to complex number."""
        real = self.xmin + px * self.x_scale
        imag = self.ymin + py * self.y_scale
        return complex(real, imag)
    
    def complex_to_pixel(self, c: complex) -> Tuple[int, int]:
        """Convert complex number to pixel coordinates."""
        px = int((c.real - self.xmin) / self.x_scale)
        py = int((c.imag - self.ymin) / self.y_scale)
        return px, py


class IterationResult:
    """Container for fractal iteration results."""
    
    def __init__(self, iterations: np.ndarray, escaped: np.ndarray,
                 final_values: Optional[np.ndarray] = None):
        """
        Initialize iteration result.
        
        Args:
            iterations: Array of iteration counts
            escaped: Boolean array indicating which points escaped
            final_values: Final complex values (for smooth coloring)
        """
        self.iterations = iterations
        self.escaped = escaped
        self.final_values = final_values
        self.shape = iterations.shape
    
    def get_normalized_iterations(self, max_iter: int) -> np.ndarray:
        """Get normalized iteration counts for smooth coloring."""
        if self.final_values is not None:
            # Continuous/smooth iteration count
            log_zn = np.log(np.abs(self.final_values) + 1e-10)
            smooth_iter = self.iterations - np.log2(np.log2(log_zn))
            smooth_iter = np.where(self.escaped, smooth_iter, max_iter)
            return np.clip(smooth_iter, 0, max_iter)
        else:
            return self.iterations.astype(np.float64)


class FractalIterator:
    """Main class for fractal iteration algorithms."""
    
    def __init__(self, max_iter: int = 1000, escape_radius: float = 2.0,
                 dtype: np.dtype = np.complex128):
        """
        Initialize fractal iterator.
        
        Args:
            max_iter: Maximum number of iterations
            escape_radius: Radius for escape condition
            dtype: Data type for calculations
        """
        self.max_iter = max_iter
        self.escape_radius = escape_radius
        self.escape_radius_sq = escape_radius ** 2
        self.dtype = dtype
        
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if escape_radius <= 0:
            raise ValueError("escape_radius must be positive")
    
    def mandelbrot_iteration(self, c: np.ndarray, z0: Optional[np.ndarray] = None) -> IterationResult:
        """
        Compute Mandelbrot set iterations.
        
        Args:
            c: Complex parameter array
            z0: Initial z values (defaults to zeros)
            
        Returns:
            IterationResult with iteration counts and escape information
        """
        if z0 is None:
            z = np.zeros_like(c, dtype=self.dtype)
        else:
            z = z0.astype(self.dtype)
        
        # Initialize result arrays
        iterations = np.zeros(c.shape, dtype=np.int32)
        escaped = np.zeros(c.shape, dtype=bool)
        
        # Main iteration loop
        for i in range(self.max_iter):
            # Check escape condition
            mask = (np.abs(z) ** 2) <= self.escape_radius_sq
            active = mask & ~escaped
            
            if not np.any(active):
                break
            
            # Update iteration counts
            iterations[active] = i
            
            # Mandelbrot iteration: z = z^2 + c
            z[active] = z[active] ** 2 + c[active]
        
        # Mark escaped points
        escaped = (np.abs(z) ** 2) > self.escape_radius_sq
        
        # Store final values for smooth coloring
        final_values = z.copy()
        
        return IterationResult(iterations, escaped, final_values)
    
    def julia_iteration(self, z: np.ndarray, c: complex) -> IterationResult:
        """
        Compute Julia set iterations.
        
        Args:
            z: Initial complex values array
            c: Julia set constant
            
        Returns:
            IterationResult with iteration counts and escape information
        """
        z = z.astype(self.dtype)
        c = self.dtype(c)
        
        # Initialize result arrays
        iterations = np.zeros(z.shape, dtype=np.int32)
        escaped = np.zeros(z.shape, dtype=bool)
        
        # Main iteration loop
        for i in range(self.max_iter):
            # Check escape condition
            mask = (np.abs(z) ** 2) <= self.escape_radius_sq
            active = mask & ~escaped
            
            if not np.any(active):
                break
            
            # Update iteration counts
            iterations[active] = i
            
            # Julia iteration: z = z^2 + c
            z[active] = z[active] ** 2 + c
        
        # Mark escaped points
        escaped = (np.abs(z) ** 2) > self.escape_radius_sq
        
        # Store final values for smooth coloring
        final_values = z.copy()
        
        return IterationResult(iterations, escaped, final_values)
    
    def burning_ship_iteration(self, c: np.ndarray) -> IterationResult:
        """
        Compute Burning Ship fractal iterations.
        
        Args:
            c: Complex parameter array
            
        Returns:
            IterationResult with iteration counts and escape information
        """
        z = np.zeros_like(c, dtype=self.dtype)
        
        # Initialize result arrays
        iterations = np.zeros(c.shape, dtype=np.int32)
        escaped = np.zeros(c.shape, dtype=bool)
        
        # Main iteration loop
        for i in range(self.max_iter):
            # Check escape condition
            mask = (np.abs(z) ** 2) <= self.escape_radius_sq
            active = mask & ~escaped
            
            if not np.any(active):
                break
            
            # Update iteration counts
            iterations[active] = i
            
            # Burning Ship iteration: z = (|Re(z)| + i|Im(z)|)^2 + c
            z_abs = np.abs(z.real) + 1j * np.abs(z.imag)
            z[active] = z_abs[active] ** 2 + c[active]
        
        # Mark escaped points
        escaped = (np.abs(z) ** 2) > self.escape_radius_sq
        
        return IterationResult(iterations, escaped, z.copy())
    
    def multibrot_iteration(self, c: np.ndarray, power: float = 2.0) -> IterationResult:
        """
        Compute Multibrot set iterations.
        
        Args:
            c: Complex parameter array
            power: Exponent for the iteration formula
            
        Returns:
            IterationResult with iteration counts and escape information
        """
        z = np.zeros_like(c, dtype=self.dtype)
        
        # Initialize result arrays
        iterations = np.zeros(c.shape, dtype=np.int32)
        escaped = np.zeros(c.shape, dtype=bool)
        
        # Main iteration loop
        for i in range(self.max_iter):
            # Check escape condition
            mask = (np.abs(z) ** 2) <= self.escape_radius_sq
            active = mask & ~escaped
            
            if not np.any(active):
                break
            
            # Update iteration counts
            iterations[active] = i
            
            # Multibrot iteration: z = z^power + c
            z[active] = z[active] ** power + c[active]
        
        # Mark escaped points
        escaped = (np.abs(z) ** 2) > self.escape_radius_sq
        
        return IterationResult(iterations, escaped, z.copy())
    
    def custom_iteration(self, initial_values: np.ndarray, 
                        iteration_func: Callable[[np.ndarray], np.ndarray]) -> IterationResult:
        """
        Compute iterations using a custom iteration function.
        
        Args:
            initial_values: Initial complex values
            iteration_func: Function that takes z array and returns new z array
            
        Returns:
            IterationResult with iteration counts and escape information
        """
        z = initial_values.astype(self.dtype)
        
        # Initialize result arrays
        iterations = np.zeros(z.shape, dtype=np.int32)
        escaped = np.zeros(z.shape, dtype=bool)
        
        # Main iteration loop
        for i in range(self.max_iter):
            # Check escape condition
            mask = (np.abs(z) ** 2) <= self.escape_radius_sq
            active = mask & ~escaped
            
            if not np.any(active):
                break
            
            # Update iteration counts
            iterations[active] = i
            
            # Apply custom iteration function
            z_new = iteration_func(z)
            z[active] = z_new[active]
        
        # Mark escaped points
        escaped = (np.abs(z) ** 2) > self.escape_radius_sq
        
        return IterationResult(iterations, escaped, z.copy())


class PeriodicityChecker:
    """Utility class for detecting periodic orbits to accelerate computation."""
    
    def __init__(self, check_period: int = 20, tolerance: float = 1e-10):
        """
        Initialize periodicity checker.
        
        Args:
            check_period: How often to check for periodicity
            tolerance: Tolerance for detecting cycles
        """
        self.check_period = check_period
        self.tolerance = tolerance
    
    def check_periodicity(self, z_history: list, current_z: complex) -> bool:
        """
        Check if the orbit has entered a periodic cycle.
        
        Args:
            z_history: List of previous z values
            current_z: Current z value
            
        Returns:
            True if periodicity is detected
        """
        if len(z_history) < 3:
            return False
        
        # Check for cycles of length 1, 2, 3, ...
        for period in range(1, min(len(z_history) // 2, 10) + 1):
            if len(z_history) >= period * 2:
                # Check if the last 'period' values repeat
                recent = z_history[-period:]
                previous = z_history[-2*period:-period]
                
                if all(abs(a - b) < self.tolerance for a, b in zip(recent, previous)):
                    return True
        
        return False
