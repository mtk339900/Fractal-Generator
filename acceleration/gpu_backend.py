"""
GPU acceleration backend using CuPy and optionally Numba CUDA.

This module provides GPU-accelerated fractal computation with graceful
fallback to CPU when GPU is not available.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from ..core.math_functions import IterationResult

logger = logging.getLogger(__name__)

# Check for GPU libraries
GPU_BACKEND = None
try:
    import cupy as cp
    GPU_BACKEND = 'cupy'
    logger.info(f"CuPy available: {cp.__version__}")
except ImportError:
    try:
        import numba.cuda as cuda
        if cuda.is_available():
            GPU_BACKEND = 'numba_cuda'
            logger.info("Numba CUDA available")
        else:
            logger.info("CUDA not available on this system")
    except ImportError:
        pass

if GPU_BACKEND is None:
    logger.warning("No GPU backend available - GPU acceleration disabled")


class GPUMemoryManager:
    """Manages GPU memory allocation and cleanup."""
    
    def __init__(self, backend='cupy'):
        """Initialize GPU memory manager."""
        self.backend = backend
        self.allocated_arrays = []
    
    def allocate(self, shape, dtype):
        """Allocate GPU memory."""
        if self.backend == 'cupy':
            import cupy as cp
            array = cp.zeros(shape, dtype=dtype)
            self.allocated_arrays.append(array)
            return array
        else:
            # Fallback to CPU
            return np.zeros(shape, dtype=dtype)
    
    def to_gpu(self, array):
        """Transfer array to GPU."""
        if self.backend == 'cupy':
            import cupy as cp
            if isinstance(array, np.ndarray):
                gpu_array = cp.asarray(array)
                self.allocated_arrays.append(gpu_array)
                return gpu_array
            return array
        else:
            return array
    
    def to_cpu(self, array):
        """Transfer array to CPU."""
        if self.backend == 'cupy':
            import cupy as cp
            if isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
            return array
        else:
            return array
    
    def cleanup(self):
        """Clean up GPU memory."""
        if self.backend == 'cupy':
            import cupy as cp
            self.allocated_arrays.clear()
            cp.get_default_memory_pool().free_all_blocks()
    
    def get_memory_info(self):
        """Get GPU memory information."""
        if self.backend == 'cupy':
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'backend': 'cupy'
            }
        return {'backend': 'cpu'}


def create_cupy_mandelbrot_kernel():
    """Create CuPy RawKernel for Mandelbrot computation."""
    if GPU_BACKEND != 'cupy':
        return None
    
    import cupy as cp
    
    kernel_code = '''
    extern "C" __global__
    void mandelbrot_kernel(const double* c_real, const double* c_imag,
                          int* iterations, bool* escaped,
                          double* final_real, double* final_imag,
                          int width, int height, int max_iter, double escape_radius_sq) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        
        if (idx >= width || idy >= height) return;
        
        int index = idy * width + idx;
        
        double cr = c_real[index];
        double ci = c_imag[index];
        
        double zr = 0.0;
        double zi = 0.0;
        
        int iter = 0;
        bool has_escaped = false;
        
        for (int n = 0; n < max_iter; n++) {
            double zr_sq = zr * zr;
            double zi_sq = zi * zi;
            
            if (zr_sq + zi_sq > escape_radius_sq) {
                iter = n;
                has_escaped = true;
                break;
            }
            
            // z = z^2 + c
            double temp = zr_sq - zi_sq + cr;
            zi = 2.0 * zr * zi + ci;
            zr = temp;
        }
        
        iterations[index] = iter;
        escaped[index] = has_escaped;
        final_real[index] = zr;
        final_imag[index] = zi;
    }
    '''
    
    return cp.RawKernel(kernel_code, 'mandelbrot_kernel')


def create_cupy_julia_kernel():
    """Create CuPy RawKernel for Julia set computation."""
    if GPU_BACKEND != 'cupy':
        return None
    
    import cupy as cp
    
    kernel_code = '''
    extern "C" __global__
    void julia_kernel(const double* z_real, const double* z_imag,
                     double c_real, double c_imag,
                     int* iterations, bool* escaped,
                     double* final_real, double* final_imag,
                     int width, int height, int max_iter, double escape_radius_sq) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int idy = blockDim.y * blockIdx.y + threadIdx.y;
        
        if (idx >= width || idy >= height) return;
        
        int index = idy * width + idx;
        
        double zr = z_real[index];
        double zi = z_imag[index];
        
        int iter = 0;
        bool has_escaped = false;
        
        for (int n = 0; n < max_iter; n++) {
            double zr_sq = zr * zr;
            double zi_sq = zi * zi;
            
            if (zr_sq + zi_sq > escape_radius_sq) {
                iter = n;
                has_escaped = true;
                break;
            }
            
            // z = z^2 + c
            double temp = zr_sq - zi_sq + c_real;
            zi = 2.0 * zr * zi + c_imag;
            zr = temp;
        }
        
        iterations[index] = iter;
        escaped[index] = has_escaped;
        final_real[index] = zr;
        final_imag[index] = zi;
    }
    '''
    
    return cp.RawKernel(kernel_code, 'julia_kernel')


class CuPyAccelerator:
    """CuPy-based GPU accelerator for fractal computation."""
    
    def __init__(self):
        """Initialize CuPy accelerator."""
        self.available = (GPU_BACKEND == 'cupy')
        self.memory_manager = GPUMemoryManager('cupy')
        
        if self.available:
            import cupy as cp
            self.device_id = cp.cuda.Device().id
            logger.info(f"Using GPU device: {self.device_id}")
            
            # Compile kernels
            self.mandelbrot_kernel = create_cupy_mandelbrot_kernel()
            self.julia_kernel = create_cupy_julia_kernel()
        else:
            logger.warning("CuPy not available - GPU acceleration disabled")
    
    def mandelbrot_iteration(self, c, max_iter, escape_radius):
        """
        GPU-accelerated Mandelbrot computation.
        
        Args:
            c: Complex parameter array
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("CuPy not available")
        
        import cupy as cp
        
        height, width = c.shape
        
        # Transfer data to GPU
        c_gpu = self.memory_manager.to_gpu(c)
        c_real_gpu = cp.ascontiguousarray(c_gpu.real.astype(cp.float64))
        c_imag_gpu = cp.ascontiguousarray(c_gpu.imag.astype(cp.float64))
        
        # Allocate output arrays on GPU
        iterations_gpu = self.memory_manager.allocate((height, width), cp.int32)
        escaped_gpu = self.memory_manager.allocate((height, width), cp.bool_)
        final_real_gpu = self.memory_manager.allocate((height, width), cp.float64)
        final_imag_gpu = self.memory_manager.allocate((height, width), cp.float64)
        
        # Launch kernel
        block_size = (16, 16)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1])
        
        self.mandelbrot_kernel(
            grid_size, block_size,
            (c_real_gpu, c_imag_gpu, iterations_gpu, escaped_gpu,
             final_real_gpu, final_imag_gpu,
             width, height, max_iter, escape_radius ** 2)
        )
        
        # Transfer results back to CPU
        iterations = self.memory_manager.to_cpu(iterations_gpu)
        escaped = self.memory_manager.to_cpu(escaped_gpu)
        final_real = self.memory_manager.to_cpu(final_real_gpu)
        final_imag = self.memory_manager.to_cpu(final_imag_gpu)
        
        final_values = final_real + 1j * final_imag
        
        return IterationResult(iterations, escaped, final_values)
    
    def julia_iteration(self, z, c, max_iter, escape_radius):
        """
        GPU-accelerated Julia set computation.
        
        Args:
            z: Initial complex values array
            c: Julia constant
            max_iter: Maximum iterations
            escape_radius: Escape radius
            
        Returns:
            IterationResult
        """
        if not self.available:
            raise RuntimeError("CuPy not available")
        
        import cupy as cp
        
        height, width = z.shape
        
        # Transfer data to GPU
        z_gpu = self.memory_manager.to_gpu(z)
        z_real_gpu = cp.ascontiguousarray(z_gpu.real.astype(cp.float64))
        z_imag_gpu = cp.ascontiguousarray(z_gpu.imag.astype(cp.float64))
        
        # Allocate output arrays
        iterations_gpu = self.memory_manager.allocate((height, width), cp.int32)
        escaped_gpu = self.memory_manager.allocate((height, width), cp.bool_)
        final_real_gpu = self.memory_manager.allocate((height, width), cp.float64)
        final_imag_gpu = self.memory_manager.allocate((height, width), cp.float64)
        
        # Launch kernel
        block_size = (16, 16)
        grid_size = ((width + block_size[0] - 1) // block_size[0],
                     (height + block_size[1] - 1) // block_size[1])
        
        self.julia_kernel(
            grid_size, block_size,
            (z_real_gpu, z_imag_gpu, float(c.real), float(c.imag),
             iterations_gpu, escaped_gpu, final_real_gpu, final_imag_gpu,
             width, height, max_iter, escape_radius ** 2)
        )
        
        # Transfer results back
        iterations = self.memory_manager.to_cpu(iterations_gpu)
        escaped = self.memory_manager.to_cpu(escaped_gpu)
        final_real = self.memory_manager.to_cpu(final_real_gpu)
        final_imag = self.memory_manager.to_cpu(final_imag_gpu)
        
        final_values = final_real + 1j * final_imag
        
        return IterationResult(iterations, escaped, final_values)
    
    def cleanup(self):
        """Clean up GPU memory."""
        self.memory_manager.cleanup()
    
    def get_device_info(self):
        """Get GPU device information."""
        if not self.available:
            return {"error": "CuPy not available"}
        
        import cupy as cp
        
        device = cp.cuda.Device()
        return {
            'device_id': device.id,
            'device_name': device.attributes['name'],
            'compute_capability': device.compute_capability,
            'total_memory': device.mem_info[1],
            'free_memory': device.mem_info[0],
            'memory_info': self.memory_manager.get_memory_info()
        }


class NumbaGPUAccelerator:
    """Numba CUDA accelerator for fractal computation."""
    
    def __init__(self):
        """Initialize Numba CUDA accelerator."""
        self.available = (GPU_BACKEND == 'numba_cuda')
        
        if self.available:
            import numba.cuda as cuda
            self.device = cuda.get_current_device()
            logger.info(f"Using CUDA device: {self.device.name}")
        else:
            logger.warning("Numba CUDA not available")
    
    def mandelbrot_iteration(self, c, max_iter, escape_radius):
        """Numba CUDA Mandelbrot computation (simplified implementation)."""
        if not self.available:
            raise RuntimeError("Numba CUDA not available")
        
        # This would require implementing CUDA kernels with Numba
        # For now, fall back to CPU
        logger.warning("Numba CUDA implementation not complete, falling back to CPU")
        raise NotImplementedError("Numba CUDA backend not fully implemented")


class GPUAccelerator:
    """Main GPU accelerator with automatic backend selection."""
    
    def __init__(self, preferred_backend=None):
        """
        Initialize GPU accelerator with backend selection.
        
        Args:
            preferred_backend: 'cupy' or 'numba_cuda', None for auto-select
        """
        self.backend = None
        self.accelerator = None
        
        # Select backend
        if preferred_backend == 'cupy' and GPU_BACKEND == 'cupy':
            self.backend = 'cupy'
            self.accelerator = CuPyAccelerator()
        elif preferred_backend == 'numba_cuda' and GPU_BACKEND == 'numba_cuda':
            self.backend = 'numba_cuda'
            self.accelerator = NumbaGPUAccelerator()
        elif GPU_BACKEND == 'cupy':
            self.backend = 'cupy'
            self.accelerator = CuPyAccelerator()
        elif GPU_BACKEND == 'numba_cuda':
            self.backend = 'numba_cuda'
            self.accelerator = NumbaGPUAccelerator()
        
        if self.backend:
            logger.info(f"GPU accelerator initialized with backend: {self.backend}")
        else:
            logger.warning("No GPU backend available")
    
    @property
    def available(self):
        """Check if GPU acceleration is available."""
        return self.accelerator is not None and self.accelerator.available
    
    def mandelbrot_iteration(self, c, max_iter, escape_radius):
        """GPU-accelerated Mandelbrot computation."""
        if not self.available:
            raise RuntimeError("GPU acceleration not available")
        return self.accelerator.mandelbrot_iteration(c, max_iter, escape_radius)
    
    def julia_iteration(self, z, c, max_iter, escape_radius):
        """GPU-accelerated Julia computation."""
        if not self.available:
            raise RuntimeError("GPU acceleration not available")
        return self.accelerator.julia_iteration(z, c, max_iter, escape_radius)
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.accelerator:
            self.accelerator.cleanup()
    
    def get_device_info(self):
        """Get GPU device information."""
        if self.accelerator:
            return self.accelerator.get_device_info()
        return {"error": "No GPU backend available"}
    
    def benchmark_performance(self, size=(1000, 1000), max_iter=1000):
        """
        Benchmark GPU performance.
        
        Args:
            size: Image size for benchmark
            max_iter: Maximum iterations
            
        Returns:
            Timing and performance information
        """
        if not self.available:
            return {"error": "GPU acceleration not available"}
        
        import time
        
        width, height = size
        x = np.linspace(-2.0, 1.0, width, dtype=np.float64)
        y = np.linspace(-1.5, 1.5, height, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        c = X + 1j * Y
        
        # Warm up
        if width * height > 10000:
            small_c = c[:100, :100]
            self.mandelbrot_iteration(small_c, 10, 2.0)
            self.cleanup()  # Clean up warm-up memory
        
        # Benchmark
        start_time = time.time()
        result = self.mandelbrot_iteration(c, max_iter, 2.0)
        gpu_time = time.time() - start_time
        
        return {
            "backend": self.backend,
            "gpu_time": gpu_time,
            "resolution": f"{width}x{height}",
            "max_iterations": max_iter,
            "pixels_per_second": (width * height) / gpu_time,
            "device_info": self.get_device_info()
        }


# Global GPU accelerator instance
_gpu_accelerator = None


def get_gpu_accelerator(preferred_backend=None):
    """Get the global GPU accelerator instance."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator(preferred_backend)
    return _gpu_accelerator


def is_gpu_available():
    """Check if GPU acceleration is available."""
    return GPU_BACKEND is not None


def get_gpu_backend():
    """Get the available GPU backend name."""
    return GPU_BACKEND


def compare_gpu_cpu_performance(size=(1000, 1000), max_iter=1000):
    """
    Compare GPU vs CPU performance for fractal computation.
    
    Args:
        size: Image size for benchmark
        max_iter: Maximum iterations
        
    Returns:
        Performance comparison dictionary
    """
    results = {
        "resolution": f"{size[0]}x{size[1]}",
        "max_iterations": max_iter,
        "total_pixels": size[0] * size[1]
    }
    
    # Create test data
    width, height = size
    x = np.linspace(-2.0, 1.0, width, dtype=np.float64)
    y = np.linspace(-1.5, 1.5, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y
    
    # Test CPU performance (NumPy baseline)
    import time
    from ..core.math_functions import FractalIterator
    
    cpu_iterator = FractalIterator(max_iter, 2.0)
    start_time = time.time()
    cpu_result = cpu_iterator.mandelbrot_iteration(c)
    cpu_time = time.time() - start_time
    
    results["cpu_time"] = cpu_time
    results["cpu_pixels_per_second"] = (width * height) / cpu_time
    
    # Test GPU performance if available
    if is_gpu_available():
        try:
            gpu_accel = get_gpu_accelerator()
            if gpu_accel.available:
                gpu_benchmark = gpu_accel.benchmark_performance(size, max_iter)
                results["gpu_time"] = gpu_benchmark["gpu_time"]
                results["gpu_pixels_per_second"] = gpu_benchmark["pixels_per_second"]
                results["gpu_backend"] = gpu_benchmark["backend"]
                results["speedup"] = cpu_time / gpu_benchmark["gpu_time"]
                results["device_info"] = gpu_benchmark["device_info"]
                
                gpu_accel.cleanup()
            else:
                results["gpu_error"] = "GPU accelerator not available"
        except Exception as e:
            results["gpu_error"] = str(e)
    else:
        results["gpu_error"] = "No GPU backend available"
    
    return results
