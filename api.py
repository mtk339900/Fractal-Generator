"""
Main API classes for fractal generation.

This module provides the high-level interface for fractal generation,
combining all the backend components into easy-to-use classes.
"""

import numpy as np
from typing import Optional, Union, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time

from .core.fractal_types import FractalType, FractalRegistry
from .core.math_functions import FractalIterator, ComplexPlane
from .core.precision import PrecisionConfig, detect_precision_need
from .rendering.coloring import ColoringEngine, ColorRGB
from .rendering.image_output import ImageExporter, RenderMetadata
from .accelerators.numba_backend import get_numba_accelerator, is_numba_available
from .accelerators.gpu_backend import get_gpu_accelerator, is_gpu_available
from .accelerators.multiprocessing import get_multiprocessing_accelerator, get_optimal_process_count
from .tools.tiling import TileRenderer
from .io.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for fractal rendering."""
    
    # Image parameters
    width: int = 1920
    height: int = 1080
    bounds: Tuple[float, float, float, float] = (-2.5, 1.0, -1.25, 1.25)  # xmin, xmax, ymin, ymax
    
    # Fractal parameters
    max_iterations: int = 1000
    escape_radius: float = 2.0
    
    # Quality and precision
    precision: str = 'double'  # 'single', 'double', 'quad', or number
    antialiasing: int = 1  # Supersampling factor
    
    # Coloring
    coloring_algorithm: str = 'smooth'
    color_palette: str = 'hot'
    inside_color: Optional[Tuple[float, float, float]] = None
    
    # Performance
    use_gpu: bool = True
    use_numba: bool = True
    use_multiprocessing: bool = True
    num_processes: Optional[int] = None
    tile_size: int = 256
    
    # Output
    output_format: str = 'png'
    jpeg_quality: int = 95
    save_metadata: bool = True
    save_raw_data: bool = False
    
    def validate(self):
        """Validate configuration parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if self.escape_radius <= 0:
            raise ValueError("escape_radius must be positive")
        
        if len(self.bounds) != 4:
            raise ValueError("bounds must be (xmin, xmax, ymin, ymax)")
        
        xmin, xmax, ymin, ymax = self.bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounds: min values must be less than max")
        
        if self.antialiasing < 1:
            raise ValueError("antialiasing must be >= 1")
        
        if self.tile_size < 32:
            raise ValueError("tile_size must be >= 32")


class FractalRenderer:
    """Main fractal rendering engine."""
    
    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize fractal renderer.
        
        Args:
            config: Rendering configuration (uses defaults if None)
        """
        self.config = config or RenderConfig()
        self.config.validate()
        
        # Initialize components
        self.coloring_engine = ColoringEngine()
        self.image_exporter = ImageExporter()
        self.precision_config = PrecisionConfig(self.config.precision)
        
        # Initialize accelerators
        self._setup_accelerators()
        
        logger.info(f"FractalRenderer initialized: {self.config.width}x{self.config.height}, "
                   f"precision={self.config.precision}")
    
    def _setup_accelerators(self):
        """Setup available acceleration backends."""
        self.accelerators = {
            'numba': None,
            'gpu': None,
            'multiprocessing': None
        }
        
        # Setup Numba
        if self.config.use_numba and is_numba_available():
            try:
                self.accelerators['numba'] = get_numba_accelerator()
                logger.info("Numba acceleration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Numba: {e}")
        
        # Setup GPU
        if self.config.use_gpu and is_gpu_available():
            try:
                self.accelerators['gpu'] = get_gpu_accelerator()
                if self.accelerators['gpu'].available:
                    logger.info(f"GPU acceleration enabled: {self.accelerators['gpu'].backend}")
                else:
                    self.accelerators['gpu'] = None
            except Exception as e:
                logger.warning(f"Failed to initialize GPU acceleration: {e}")
        
        # Setup multiprocessing
        if self.config.use_multiprocessing:
            try:
                num_proc = self.config.num_processes or get_optimal_process_count()
                self.accelerators['multiprocessing'] = get_multiprocessing_accelerator(
                    num_proc, self.config.tile_size
                )
                logger.info(f"Multiprocessing enabled: {num_proc} processes")
            except Exception as e:
                logger.warning(f"Failed to initialize multiprocessing: {e}")
    
    def render(self, fractal: FractalType, output_path: Optional[Path] = None,
               progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Render fractal to image.
        
        Args:
            fractal: Fractal type to render
            output_path: Optional output file path
            progress_callback: Optional progress callback function
            
        Returns:
            RGB image array (0-1 range)
        """
        start_time = time.time()
        
        logger.info(f"Starting render: {fractal.name} fractal")
        
        # Create complex plane
        plane = ComplexPlane(*self.config.bounds, self.config.width, self.config.height)
        
        # Create iterator with appropriate precision
        iterator = FractalIterator(
            max_iter=self.config.max_iterations,
            escape_radius=self.config.escape_radius,
            dtype=self.precision_config.dtype or np.complex128
        )
        
        # Choose rendering method based on available accelerators and problem size
        result = self._render_with_best_method(fractal, plane, iterator, progress_callback)
        
        # Apply antialiasing if requested
        if self.config.antialiasing > 1:
            result = self._apply_antialiasing(fractal, plane, iterator, self.config.antialiasing)
        
        # Generate colored image
        inside_color = None
        if self.config.inside_color:
            inside_color = ColorRGB(*self.config.inside_color)
        
        rgb_image = self.coloring_engine.render_color_image(
            result,
            algorithm=self.config.coloring_algorithm,
            palette=self.config.color_palette,
            max_iter=self.config.max_iterations,
            inside_color=inside_color
        )
        
        # Save image if path provided
        if output_path:
            self._save_image(rgb_image, output_path, time.time() - start_time, fractal)
        
        total_time = time.time() - start_time
        logger.info(f"Render complete: {total_time:.2f}s")
        
        return rgb_image
    
    def _render_with_best_method(self, fractal: FractalType, plane: ComplexPlane,
                                iterator: FractalIterator, progress_callback) -> Any:
        """Choose and execute the best rendering method."""
        total_pixels = plane.width * plane.height
        
        # For very high precision, use arbitrary precision backend
        if self.precision_config.use_mpmath:
            logger.info("Using arbitrary precision rendering")
            return self._render_high_precision(fractal, plane, iterator)
        
        # For GPU-suitable problems, try GPU first
        if (self.accelerators['gpu'] and 
            total_pixels >= 500000 and  # Minimum size for GPU efficiency
            not self.precision_config.use_mpmath):
            try:
                logger.info("Attempting GPU-accelerated render")
                return self._render_gpu(fractal, plane, iterator)
            except Exception as e:
                logger.warning(f"GPU render failed, falling back: {e}")
        
        # For large images, use multiprocessing
        if (self.accelerators['multiprocessing'] and 
            total_pixels >= 1000000):  # 1M+ pixels
            logger.info("Using multiprocessing render")
            return self._render_multiprocessing(fractal, plane, iterator, progress_callback)
        
        # For medium images, try Numba
        if self.accelerators['numba']:
            logger.info("Using Numba-accelerated render")
            return self._render_numba(fractal, plane, iterator)
        
        # Fallback to basic rendering
        logger.info("Using basic CPU render")
        return fractal.compute(plane, iterator)
    
    def _render_gpu(self, fractal: FractalType, plane: ComplexPlane, iterator: FractalIterator):
        """Render using GPU acceleration."""
        gpu_accel = self.accelerators['gpu']
        
        if fractal.name.lower() == 'mandelbrot':
            c = plane.create_complex_array(iterator.dtype)
            return gpu_accel.mandelbrot_iteration(c, iterator.max_iter, iterator.escape_radius)
        elif fractal.name.lower() == 'julia':
            z = plane.create_complex_array(iterator.dtype)
            return gpu_accel.julia_iteration(z, fractal.parameters.c, 
                                           iterator.max_iter, iterator.escape_radius)
        else:
            raise NotImplementedError(f"GPU rendering not implemented for {fractal.name}")
    
    def _render_numba(self, fractal: FractalType, plane: ComplexPlane, iterator: FractalIterator):
        """Render using Numba acceleration."""
        numba_accel = self.accelerators['numba']
        
        if fractal.name.lower() == 'mandelbrot':
            c = plane.create_complex_array(iterator.dtype)
            return numba_accel.mandelbrot_iteration(c, iterator.max_iter, iterator.escape_radius)
        elif fractal.name.lower() == 'julia':
            z = plane.create_complex_array(iterator.dtype)
            return numba_accel.julia_iteration(z, fractal.parameters.c,
                                             iterator.max_iter, iterator.escape_radius)
        else:
            # Fall back to standard computation for unsupported types
            return fractal.compute(plane, iterator)
    
    def _render_multiprocessing(self, fractal: FractalType, plane: ComplexPlane,
                               iterator: FractalIterator, progress_callback):
        """Render using multiprocessing acceleration."""
        mp_accel = self.accelerators['multiprocessing']
        
        if progress_callback:
            # Use progressive renderer for progress updates
            from .accelerators.multiprocessing import ProgressiveRenderer
            progressive = ProgressiveRenderer(mp_accel.num_processes, mp_accel.tile_size)
            
            def tile_progress(completed, total, image):
                progress = completed / total
                progress_callback(progress, image)
            
            progressive.add_progress_callback(tile_progress)
            return progressive.render_progressive(fractal, plane, iterator)
        else:
            return mp_accel.render_fractal_parallel(fractal, plane, iterator)
    
    def _render_high_precision(self, fractal: FractalType, plane: ComplexPlane, iterator: FractalIterator):
        """Render using arbitrary precision arithmetic."""
        from .core.precision import ArbitraryPrecisionPlane, HighPrecisionIterator, HighPrecisionComplex
        
        # Convert to high-precision plane
        hp_plane = ArbitraryPrecisionPlane(
            str(plane.xmin), str(plane.xmax),
            str(plane.ymin), str(plane.ymax),
            plane.width, plane.height
        )
        
        hp_iterator = HighPrecisionIterator(iterator.max_iter, str(iterator.escape_radius))
        
        if fractal.name.lower() == 'mandelbrot':
            iterations, converged = hp_iterator.render_mandelbrot_region(hp_plane)
            # Convert back to standard format
            escaped = ~converged
            return type('IterationResult', (), {
                'iterations': iterations,
                'escaped': escaped,
                'final_values': None,
                'shape': iterations.shape
            })()
        else:
            raise NotImplementedError(f"High-precision rendering not implemented for {fractal.name}")
    
    def _apply_antialiasing(self, fractal: FractalType, plane: ComplexPlane, 
                          iterator: FractalIterator, aa_factor: int):
        """Apply antialiasing by supersampling."""
        if aa_factor <= 1:
            return fractal.compute(plane, iterator)
        
        logger.info(f"Applying {aa_factor}x antialiasing")
        
        # Create supersampled plane
        aa_width = plane.width * aa_factor
        aa_height = plane.height * aa_factor
        aa_plane = ComplexPlane(plane.xmin, plane.xmax, plane.ymin, plane.ymax,
                               aa_width, aa_height)
        
        # Render at high resolution
        aa_result = self._render_with_best_method(fractal, aa_plane, iterator, None)
        
        # Downsample by averaging
        from scipy.ndimage import zoom
        
        downsample_factor = 1.0 / aa_factor
        iterations = zoom(aa_result.iterations.astype(np.float32), downsample_factor, order=1)
        escaped = zoom(aa_result.escaped.astype(np.float32), downsample_factor, order=0) > 0.5
        
        final_values = None
        if aa_result.final_values is not None:
            # Handle complex values separately
            real_part = zoom(aa_result.final_values.real, downsample_factor, order=1)
            imag_part = zoom(aa_result.final_values.imag, downsample_factor, order=1)
            final_values = real_part + 1j * imag_part
        
        from .core.math_functions import IterationResult
        return IterationResult(iterations.astype(np.int32), escaped, final_values)
    
    def _save_image(self, rgb_image: np.ndarray, output_path: Path, render_time: float, fractal: FractalType):
        """Save rendered image with metadata."""
        output_path = Path(output_path)
        
    def _save_image(self, rgb_image: np.ndarray, output_path: Path, render_time: float, fractal: FractalType):
        """Save rendered image with metadata."""
        output_path = Path(output_path)
        
        # Create metadata
        metadata = RenderMetadata(
            fractal_type=fractal.name,
            bounds=self.config.bounds,
            resolution=(self.config.width, self.config.height),
            max_iterations=self.config.max_iterations,
            escape_radius=self.config.escape_radius,
            coloring_algorithm=self.config.coloring_algorithm,
            color_palette=self.config.color_palette,
            precision=self.config.precision,
            render_time_seconds=render_time,
            tiles_used=self.config.use_multiprocessing,
            gpu_accelerated=bool(self.accelerators['gpu']),
            fractal_parameters=fractal.parameters.to_dict() if hasattr(fractal, 'parameters') else {}
        )
        
        # Save image
        if self.config.save_metadata:
            self.image_exporter.save_image(rgb_image, output_path, metadata, self.config.jpeg_quality)
        else:
            self.image_exporter.save_image(rgb_image, output_path, quality=self.config.jpeg_quality)
        
        # Save raw data if requested
        if self.config.save_raw_data:
            raw_path = output_path.with_suffix('.npy')
            self.image_exporter.save_raw_data(rgb_image, raw_path, metadata)
    
    def benchmark_performance(self) -> Dict[str, Any]:
        """
        Benchmark rendering performance with current configuration.
        
        Returns:
            Performance benchmark results
        """
        from .core.fractal_types import MandelbrotSet
        
        logger.info("Starting performance benchmark")
        
        # Create test fractal
        fractal = MandelbrotSet()
        
        results = {
            'config': {
                'resolution': f"{self.config.width}x{self.config.height}",
                'max_iterations': self.config.max_iterations,
                'precision': self.config.precision,
                'accelerators_enabled': {
                    'numba': self.accelerators['numba'] is not None,
                    'gpu': self.accelerators['gpu'] is not None,
                    'multiprocessing': self.accelerators['multiprocessing'] is not None,
                }
            },
            'benchmarks': {}
        }
        
        # Test different rendering methods
        plane = ComplexPlane(*self.config.bounds, self.config.width, self.config.height)
        iterator = FractalIterator(self.config.max_iterations, self.config.escape_radius)
        
        # Baseline CPU
        start_time = time.time()
        cpu_result = fractal.compute(plane, iterator)
        cpu_time = time.time() - start_time
        
        results['benchmarks']['cpu'] = {
            'time': cpu_time,
            'pixels_per_second': (self.config.width * self.config.height) / cpu_time
        }
        
        # Test Numba if available
        if self.accelerators['numba']:
            try:
                start_time = time.time()
                self._render_numba(fractal, plane, iterator)
                numba_time = time.time() - start_time
                
                results['benchmarks']['numba'] = {
                    'time': numba_time,
                    'pixels_per_second': (self.config.width * self.config.height) / numba_time,
                    'speedup': cpu_time / numba_time
                }
            except Exception as e:
                results['benchmarks']['numba'] = {'error': str(e)}
        
        # Test GPU if available
        if self.accelerators['gpu']:
            try:
                gpu_benchmark = self.accelerators['gpu'].benchmark_performance(
                    (self.config.width, self.config.height), self.config.max_iterations
                )
                results['benchmarks']['gpu'] = gpu_benchmark
                results['benchmarks']['gpu']['speedup'] = cpu_time / gpu_benchmark['gpu_time']
            except Exception as e:
                results['benchmarks']['gpu'] = {'error': str(e)}
        
        # Test multiprocessing if available
        if self.accelerators['multiprocessing']:
            try:
                mp_benchmark = self.accelerators['multiprocessing'].benchmark_parallel_performance(
                    (self.config.width, self.config.height), self.config.max_iterations
                )
                results['benchmarks']['multiprocessing'] = mp_benchmark
            except Exception as e:
                results['benchmarks']['multiprocessing'] = {'error': str(e)}
        
        return results
    
    def update_config(self, **kwargs):
        """Update rendering configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        self.config.validate()
        
        # Reinitialize accelerators if needed
        if any(key in ['use_gpu', 'use_numba', 'use_multiprocessing', 'num_processes'] 
               for key in kwargs):
            self._setup_accelerators()


class BatchRenderer:
    """Batch fractal rendering with job queuing."""
    
    def __init__(self, base_config: Optional[RenderConfig] = None):
        """Initialize batch renderer."""
        self.base_config = base_config or RenderConfig()
        self.jobs = []
        self.results = []
    
    def add_job(self, fractal: FractalType, output_path: Path, 
               config_overrides: Optional[Dict[str, Any]] = None,
               job_name: Optional[str] = None):
        """
        Add a rendering job to the batch.
        
        Args:
            fractal: Fractal to render
            output_path: Output file path
            config_overrides: Configuration overrides for this job
            job_name: Optional name for the job
        """
        job = {
            'fractal': fractal,
            'output_path': Path(output_path),
            'config_overrides': config_overrides or {},
            'job_name': job_name or f"job_{len(self.jobs)}",
            'status': 'pending'
        }
        self.jobs.append(job)
    
    def run_batch(self, progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Execute all jobs in the batch.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of job results
        """
        results = []
        
        for i, job in enumerate(self.jobs):
            logger.info(f"Processing job {i+1}/{len(self.jobs)}: {job['job_name']}")
            
            try:
                # Create config for this job
                config = RenderConfig(**{
                    **self.base_config.__dict__,
                    **job['config_overrides']
                })
                
                # Create renderer
                renderer = FractalRenderer(config)
                
                # Render
                start_time = time.time()
                rgb_image = renderer.render(job['fractal'], job['output_path'])
                render_time = time.time() - start_time
                
                # Record result
                result = {
                    'job_name': job['job_name'],
                    'status': 'completed',
                    'render_time': render_time,
                    'output_path': str(job['output_path']),
                    'config': config.__dict__.copy()
                }
                
                job['status'] = 'completed'
                
            except Exception as e:
                logger.error(f"Job {job['job_name']} failed: {e}")
                result = {
                    'job_name': job['job_name'],
                    'status': 'failed',
                    'error': str(e)
                }
                job['status'] = 'failed'
            
            results.append(result)
            
            # Call progress callback
            if progress_callback:
                progress_callback(i + 1, len(self.jobs), result)
        
        self.results = results
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary."""
        if not self.results:
            return {'status': 'not_run'}
        
        completed = sum(1 for r in self.results if r['status'] == 'completed')
        failed = sum(1 for r in self.results if r['status'] == 'failed')
        total_time = sum(r.get('render_time', 0) for r in self.results)
        
        return {
            'total_jobs': len(self.results),
            'completed': completed,
            'failed': failed,
            'success_rate': completed / len(self.results) if self.results else 0,
            'total_render_time': total_time,
            'average_render_time': total_time / completed if completed > 0 else 0
        }


class FractalExplorer:
    """Interactive fractal exploration with zoom and parameter adjustment."""
    
    def __init__(self, initial_config: Optional[RenderConfig] = None):
        """Initialize fractal explorer."""
        self.config = initial_config or RenderConfig(width=800, height=800)
        self.renderer = FractalRenderer(self.config)
        self.history = []
        self.current_fractal = None
        self.current_image = None
    
    def set_fractal(self, fractal: FractalType):
        """Set the current fractal type."""
        self.current_fractal = fractal
        self.history = []  # Reset history when changing fractal type
    
    def render_current(self) -> np.ndarray:
        """Render current fractal with current settings."""
        if self.current_fractal is None:
            raise ValueError("No fractal set. Use set_fractal() first.")
        
        self.current_image = self.renderer.render(self.current_fractal)
        return self.current_image
    
    def zoom_to_point(self, x: int, y: int, zoom_factor: float = 2.0):
        """
        Zoom into a specific point in the image.
        
        Args:
            x, y: Pixel coordinates to zoom into
            zoom_factor: Zoom multiplication factor
        """
        if self.current_image is None:
            raise ValueError("No current image. Render first.")
        
        # Save current state to history
        self.history.append({
            'bounds': self.config.bounds,
            'max_iterations': self.config.max_iterations
        })
        
        # Convert pixel coordinates to complex plane
        plane = ComplexPlane(*self.config.bounds, self.config.width, self.config.height)
        center = plane.pixel_to_complex(x, y)
        
        # Calculate new bounds
        current_width = self.config.bounds[1] - self.config.bounds[0]
        current_height = self.config.bounds[3] - self.config.bounds[2]
        
        new_width = current_width / zoom_factor
        new_height = current_height / zoom_factor
        
        new_bounds = (
            center.real - new_width / 2,
            center.real + new_width / 2,
            center.imag - new_height / 2,
            center.imag + new_height / 2
        )
        
        # Update configuration
        self.config.bounds = new_bounds
        
        # Increase iterations for deeper zoom if needed
        zoom_level = 1.0 / (new_width * new_height)
        if zoom_level > 1000:
            recommended_iter = min(10000, int(self.config.max_iterations * 1.5))
            self.config.max_iterations = recommended_iter
        
        # Update precision if needed for very deep zooms
        precision_needed = detect_precision_need(zoom_factor, center.real, center.imag)
        if precision_needed != self.config.precision:
            logger.info(f"Updating precision to {precision_needed} for deep zoom")
            self.config.precision = precision_needed
            # Recreate renderer with new precision
            self.renderer = FractalRenderer(self.config)
        
        logger.info(f"Zoomed to {center} with factor {zoom_factor}")
    
    def zoom_out(self, zoom_factor: float = 0.5):
        """Zoom out from current view."""
        current_width = self.config.bounds[1] - self.config.bounds[0]
        current_height = self.config.bounds[3] - self.config.bounds[2]
        
        center_real = (self.config.bounds[0] + self.config.bounds[1]) / 2
        center_imag = (self.config.bounds[2] + self.config.bounds[3]) / 2
        
        new_width = current_width / zoom_factor
        new_height = current_height / zoom_factor
        
        new_bounds = (
            center_real - new_width / 2,
            center_real + new_width / 2,
            center_imag - new_height / 2,
            center_imag + new_height / 2
        )
        
        self.config.bounds = new_bounds
        logger.info(f"Zoomed out with factor {zoom_factor}")
    
    def pan(self, dx: float, dy: float):
        """
        Pan the view by the specified amount.
        
        Args:
            dx, dy: Pan amounts in complex plane units
        """
        xmin, xmax, ymin, ymax = self.config.bounds
        self.config.bounds = (xmin + dx, xmax + dx, ymin + dy, ymax + dy)
    
    def go_back(self):
        """Return to previous view from history."""
        if not self.history:
            logger.warning("No history available")
            return
        
        previous_state = self.history.pop()
        self.config.bounds = previous_state['bounds']
        self.config.max_iterations = previous_state['max_iterations']
        
        logger.info("Returned to previous view")
    
    def reset_view(self):
        """Reset to the default view for current fractal."""
        if self.current_fractal is None:
            return
        
        recommended_bounds = self.current_fractal.get_recommended_bounds()
        self.config.bounds = recommended_bounds
        self.config.max_iterations = 1000  # Reset to default
        self.config.precision = 'double'  # Reset precision
        
        # Recreate renderer
        self.renderer = FractalRenderer(self.config)
        
        self.history = []  # Clear history
        logger.info("Reset to default view")
    
    def adjust_iterations(self, new_iterations: int):
        """Adjust maximum iterations."""
        if new_iterations <= 0:
            raise ValueError("Iterations must be positive")
        
        self.config.max_iterations = new_iterations
        logger.info(f"Set max iterations to {new_iterations}")
    
    def change_palette(self, palette_name: str):
        """Change color palette."""
        available_palettes = self.renderer.coloring_engine.list_palettes()
        if palette_name not in available_palettes:
            raise ValueError(f"Unknown palette '{palette_name}'. Available: {available_palettes}")
        
        self.config.color_palette = palette_name
        logger.info(f"Changed palette to {palette_name}")
    
    def get_exploration_info(self) -> Dict[str, Any]:
        """Get current exploration state information."""
        if self.current_fractal is None:
            return {'error': 'No fractal set'}
        
        current_width = self.config.bounds[1] - self.config.bounds[0]
        current_height = self.config.bounds[3] - self.config.bounds[2]
        zoom_level = 1.0 / (current_width * current_height)
        
        center = (
            (self.config.bounds[0] + self.config.bounds[1]) / 2,
            (self.config.bounds[2] + self.config.bounds[3]) / 2
        )
        
        return {
            'fractal': self.current_fractal.name,
            'bounds': self.config.bounds,
            'center': center,
            'zoom_level': zoom_level,
            'max_iterations': self.config.max_iterations,
            'precision': self.config.precision,
            'palette': self.config.color_palette,
            'history_depth': len(self.history),
            'estimated_pixels_per_unit': self.config.width / current_width
        }
