"""
Numba JIT compilation backend for high-performance fractal computation.

This module provides JIT-compiled versions of fractal iteration functions
using Numba for significant performance improvements over pure NumPy.
"""

import numpy as np
from typing import Tuple, Optional, Callable
import logging
from ..core.math_functions import IterationResult

logger = logging.getLogger(__name__)

# Check for Numba availability
try:
    import numba
    from numba import jit, prange, types
    NUMBA_AVAILABLE = True
    logger.info(f"Numba available: {numba.__version__}")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available - JIT acceleration disabled")
    
    # Create dummy decorator for when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@jit(nopython=True, parallel=True, cache=True)
def mandelbrot_kernel(c_real, c_imag, max_iter, escape_radius_sq):
    """
    JIT-compiled Mandelbrot kernel for maximum performance.
    
    Args:
        c_real: Real components of c values
        c_imag: Imaginary components of c values  
        max_iter: Maximum iterations
        escape_radius_sq: Squared escape radius
        
    Returns:
        Tuple of (iterations, escaped, final_real, final_imag)
    """
    height, width = c_real.shape
    iterations = np.zeros((height, width), dtype=np.int32)
    escaped = np.zeros((height, width), dtype=np.bool_)
    final_real = np.zeros((height, width), dtype=np.float64)
    final_imag = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in prange(width):
            cr = c_real[i, j]
            ci = c_imag[i, j]
            
            zr = 0.0
            zi = 0.0
            
            for n in range(max_iter):
                zr_sq = zr * zr
                zi_sq = zi * zi
                
                if zr_sq + zi_sq > escape_radius_sq:
                    iterations[i, j] = n
                    escaped[i, j] = True
                    break
                
                # z = z^2 + c
                zi = 2.0 * zr * zi + ci
                zr = zr_sq - zi_sq + cr
            
            final_real[i, j] = zr
            final_imag[i, j] = zi
    
    return iterations, escaped, final_real, final_imag


@jit(nopython=True, parallel=True, cache=True)
def julia_kernel(z_real, z_imag, c_real, c_imag, max_iter, escape_radius_sq):
    """
    JIT-compiled Julia set kernel.
    
    Args:
        z_real: Real components of initial z values
        z_imag: Imaginary components of initial z values
        c_real: Real component of Julia constant
        c_imag: Imaginary component of Julia constant
        max_iter: Maximum iterations
        escape_radius_sq: Squared escape radius
        
    Returns:
        Tuple of (iterations, escaped, final_real, final_imag)
    """
    height, width = z_real.shape
    iterations = np.zeros((height, width), dtype=np.int32)
    escaped = np.zeros((height, width), dtype=np.bool_)
    final_real = np.zeros((height, width), dtype=np.float64)
    final_imag = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in prange(width):
            zr = z_real[i, j]
            zi = z_imag[i, j]
            
            for n in range(max_iter):
                zr_sq = zr * zr
                zi_sq = zi * zi
                
                if zr_sq + zi_sq > escape_radius_sq:
                    iterations[i, j] = n
                    escaped[i, j] = True
                    break
                
                # z = z^2 + c
                zi = 2.0 * zr * zi + c_imag
                zr = zr_sq - zi_sq + c_real
            
            final_real[i, j] = zr
            final_imag[i, j] = zi
    
    return iterations, escaped, final_real, final_imag


@jit(nopython=True, parallel=True, cache=True)
def burning_ship_kernel(c_real, c_imag, max_iter, escape_radius_sq):
    """
    JIT-compiled Burning Ship fractal kernel.
    
    Args:
        c_real: Real components of c values
        c_imag: Imaginary components of c values
        max_iter: Maximum iterations
        escape_radius_sq: Squared escape radius
        
    Returns:
        Tuple of (iterations, escaped, final_real, final_imag)
    """
    height, width = c_real.shape
    iterations = np.zeros((height, width), dtype=np.int32)
    escaped = np.zeros((height, width), dtype=np.bool_)
    final_real = np.zeros((height, width), dtype=np.float64)
    final_imag = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in prange(width):
            cr = c_real[i, j]
            ci = c_imag[i, j]
            
            zr = 0.0
            zi = 0.0
            
            for n in range(max_iter):
                zr_sq = zr * zr
                zi_sq = zi * zi
                
                if zr_sq + zi_sq > escape_radius_sq:
                    iterations[i, j] = n
                    escaped[i, j] = True
                    break
                
                # Burning Ship: z = (|Re(z)| + i|Im(z)|)^2 + c
                zr_abs = abs(zr)
                zi_abs = abs(zi)
                zi = 2.0 * zr_abs * zi_abs + ci
                zr = zr_abs * zr_abs - zi_abs * zi_abs + cr
            
            final_real[i, j] = zr
            final_imag[i, j] = zi
    
    return iterations, escaped, final_real, final_imag


@jit(nopython=True, parallel=True, cache=True)
def multibrot_kernel(c_real, c_imag, power, max_iter, escape_radius_sq):
    """
    JIT-compiled Multibrot fractal kernel.
    
    Args:
        c_real: Real components of c values
        c_imag: Imaginary components of c values
        power: Exponent for the iteration
        max_iter: Maximum iterations
        escape_radius_sq: Squared escape radius
        
    Returns:
        Tuple of (iterations, escaped, final_real, final_imag)
    """
    height, width = c_real.shape
    iterations = np.zeros((height, width), dtype=np.int32)
    escaped = np.zeros((height, width), dtype=np.bool_)
    final_real = np.zeros((height, width), dtype=np.float64)
    final_imag = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in prange(width):
            cr = c_real[i, j]
            ci = c_imag[i, j]
            
            zr = 0.0
            zi = 0.0
            
            for n in range(max_iter):
                zr_sq = zr * zr
                zi_sq = zi * zi
                
                if zr_sq + zi_sq > escape_radius_sq:
                    iterations[i, j] = n
                    escaped[i, j] = True
                    break
                
                # z = z^power + c (simplified for integer powers)
                if power == 2.0:
                    zi = 2.0 * zr * zi + ci
                    zr = zr_sq - zi_sq + cr
                elif power == 3.0:
                    # z^3 = (a+bi)^3 = a^3 - 3ab^2 + i(3a^2b - b^3)
                    new_zr = zr * zr * zr - 3.0 * zr * zi * zi + cr
                    new_zi = 3.0 * zr * zr * zi - zi * zi * zi + ci
                    zr = new_zr
                    zi = new_zi
                elif power == 4.0:
                    # z^4 using (z^2)^2
                    zr2 = zr_sq - zi_sq
                    zi2 = 2.0 * zr * zi
                    zr = zr2 * zr2 - zi2 * zi2 + cr
                    zi = 2.0 * zr2 * zi2 + ci
                else:
                    # General power (slower)
                    r = (zr_sq + zi_sq) ** (power / 2.0)
                    theta = power * np.arctan2(zi, zr)
                    zr = r * np.cos(theta) + cr
                    zi = r * np.sin(theta) + ci
            
            final_real[i, j] = zr
            final_imag[i, j] = zi
    
    return iterations, escaped, final_real, final_imag


@jit(nopython=True, parallel=True, cache=True)
def smooth_iteration_count(iterations, escaped, final_real, final_imag, escape_radius):
    """
    Calculate smooth iteration counts for continuous coloring.
    
    Args:
        iterations: Raw iteration counts
        escaped: Escape flags
        final_real: Final real values
        final_imag: Final imaginary values
        escape_radius: Escape radius
        
    Returns:
        Smooth iteration counts
    """
    height, width = iterations.shape
    smooth_iter = np.zeros((height, width), dtype=np.float64)
    
    for i in prange(height):
        for j in prange(width):
            if escaped[i, j]:
                # Calculate smooth iteration count
                zn_mag = np.sqrt(final_real[i, j]**2 + final_imag[i, j]**2)
                if zn_mag > 0:
                    smooth_iter[i, j] = iterations[i, j] - np.log2(np.log(zn_mag) / np.log(escape_radius))
                else:
                    smooth_iter[i, j] = iterations[i, j]
            else:
                smooth_iter[i, j] = iterations[i, j]
    
    return smooth_iter


class NumbaAccelerator:
    """Numba-accelerated fractal computation backend."""
    
    def __init__(self):
        """Initialize Numba accelerator."""
        self.available = NUMBA_AVAILABLE
        if not self.available:
            logger.warning("Numba not available - acceleration disabled")
    
    def mandelbrot_iteration(self, c, max_iter, escape_radius):
        """
        Accelerated Mandelbrot computation.
        
        Args:
            c: Complex parameter array
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("Numba not available")
        
        # Separate real and imaginary parts for Numba
        c_real = c.real.astype(np.float64)
        c_imag = c.imag.astype(np.float64)
        escape_radius_sq = escape_radius ** 2
        
        # Run JIT-compiled kernel
        iterations, escaped, final_real, final_imag = mandelbrot_kernel(
            c_real, c_imag, max_iter, escape_radius_sq
        )
        
        # Combine final values back into complex array
        final_values = final_real + 1j * final_imag
        
        return IterationResult(iterations, escaped, final_values)
    
    def julia_iteration(self, z, c, max_iter, escape_radius):
        """
        Accelerated Julia set computation.
        
        Args:
            z: Initial complex values array
            c: Julia constant
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("Numba not available")
        
        # Prepare arrays for Numba
        z_real = z.real.astype(np.float64)
        z_imag = z.imag.astype(np.float64)
        c_real = float(c.real)
        c_imag = float(c.imag)
        escape_radius_sq = escape_radius ** 2
        
        # Run JIT-compiled kernel
        iterations, escaped, final_real, final_imag = julia_kernel(
            z_real, z_imag, c_real, c_imag, max_iter, escape_radius_sq
        )
        
        # Combine final values
        final_values = final_real + 1j * final_imag
        
        return IterationResult(iterations, escaped, final_values)
    
    def burning_ship_iteration(self, c, max_iter, escape_radius):
        """
        Accelerated Burning Ship computation.
        
        Args:
            c: Complex parameter array
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("Numba not available")
        
        # Prepare arrays
        c_real = c.real.astype(np.float64)
        c_imag = c.imag.astype(np.float64)
        escape_radius_sq = escape_radius ** 2
        
        # Run kernel
        iterations, escaped, final_real, final_imag = burning_ship_kernel(
            c_real, c_imag, max_iter, escape_radius_sq
        )
        
        final_values = final_real + 1j * final_imag
        return IterationResult(iterations, escaped, final_values)
    
    def multibrot_iteration(self, c, power, max_iter, escape_radius):
        """
        Accelerated Multibrot computation.
        
        Args:
            c: Complex parameter array
            power: Exponent
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("Numba not available")
        
        # Prepare arrays
        c_real = c.real.astype(np.float64)
        c_imag = c.imag.astype(np.float64)
        escape_radius_sq = escape_radius ** 2
        
        # Run kernel
        iterations, escaped, final_real, final_imag = multibrot_kernel(
            c_real, c_imag, float(power), max_iter, escape_radius_sq
        )
        
        final_values = final_real + 1j * final_imag
        return IterationResult(iterations, escaped, final_values)
    
    def calculate_smooth_iterations(self, result, escape_radius):
        """
        Calculate smooth iteration counts using Numba acceleration.
        
        Args:
            result: IterationResult from fractal computation
            escape_radius: Escape radius used
            
        Returns:
            Smooth iteration count array
        """
        if not self.available or result.final_values is None:
            return result.iterations.astype(np.float64)
        
        return smooth_iteration_count(
            result.iterations,
            result.escaped,
            result.final_values.real,
            result.final_values.imag,
            escape_radius
        )
    
    def benchmark_performance(self, size=(1000, 1000), max_iter=1000):
        """
        Benchmark Numba performance vs pure NumPy.
        
        Args:
            size: Image size for benchmark
            max_iter: Maximum iterations
            
        Returns:
            Dictionary with timing results
        """
        if not self.available:
            return {"error": "Numba not available"}
        
        import time
        
        # Create test data
        width, height = size
        x = np.linspace(-2.0, 1.0, width)
        y = np.linspace(-1.5, 1.5, height)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        
        # Warm up JIT compiler
        small_c = c[:100, :100]
        self.mandelbrot_iteration(small_c, 10, 2.0)
        
        # Benchmark Numba version
        start_time = time.time()
        result_numba = self.mandelbrot_iteration(c, max_iter, 2.0)
        numba_time = time.time() - start_time
        
        return {
            "numba_time": numba_time,
            "resolution": f"{width}x{height}",
            "max_iterations": max_iter,
            "pixels_per_second": (width * height) / numba_time,
        }


# Global accelerator instance
_numba_accelerator = None


def get_numba_accelerator():
    """Get the global Numba accelerator instance."""
    global _numba_accelerator
    if _numba_accelerator is None:
        _numba_accelerator = NumbaAccelerator()
    return _numba_accelerator


def is_numba_available():
    """Check if Numba acceleration is available."""
    return NUMBA_AVAILABLE
