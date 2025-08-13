"""
Multiprocessing backend for parallel fractal computation.

This module provides tile-based parallel rendering using Python's
multiprocessing library for CPU-based acceleration across multiple cores.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process, shared_memory
import logging
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from ..core.math_functions import IterationResult, ComplexPlane, FractalIterator
from ..core.fractal_types import FractalType

logger = logging.getLogger(__name__)


@dataclass
class TileSpec:
    """Specification for a single tile in parallel rendering."""
    tile_id: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    width: int
    height: int
    
    def get_bounds(self, plane: ComplexPlane) -> Tuple[float, float, float, float]:
        """Get complex plane bounds for this tile."""
        xmin = plane.xmin + self.x_start * plane.x_scale
        xmax = plane.xmin + self.x_end * plane.x_scale
        ymin = plane.ymin + self.y_start * plane.y_scale
        ymax = plane.ymin + self.y_end * plane.y_scale
        return xmin, xmax, ymin, ymax


@dataclass
class TileResult:
    """Result from processing a single tile."""
    tile_id: int
    iterations: np.ndarray
    escaped: np.ndarray
    final_values: Optional[np.ndarray]
    x_start: int
    y_start: int
    processing_time: float


def create_tile_grid(width: int, height: int, tile_size: int = 256) -> List[TileSpec]:
    """
    Create a grid of tiles for parallel processing.
    
    Args:
        width: Total image width
        height: Total image height
        tile_size: Target tile size (pixels)
        
    Returns:
        List of TileSpec objects
    """
    tiles = []
    tile_id = 0
    
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            
            tile = TileSpec(
                tile_id=tile_id,
                x_start=x,
                x_end=x_end,
                y_start=y,
                y_end=y_end,
                width=x_end - x,
                height=y_end - y
            )
            tiles.append(tile)
            tile_id += 1
    
    logger.info(f"Created {len(tiles)} tiles of target size {tile_size}x{tile_size}")
    return tiles


def process_fractal_tile(args):
    """
    Process a single fractal tile in a separate process.
    
    Args:
        args: Tuple of (fractal_params, tile_spec, iterator_params, plane_params)
        
    Returns:
        TileResult object
    """
    try:
        fractal_type, fractal_params, tile_spec, iterator_params, plane_params = args
        
        start_time = time.time()
        
        # Reconstruct objects in worker process
        from ..core.fractal_types import FractalRegistry
        
        # Create fractal instance
        fractal = FractalRegistry.create_fractal(fractal_type, **fractal_params)
        
        # Create iterator
        iterator = FractalIterator(**iterator_params)
        
        # Create tile plane
        tile_bounds = tile_spec.get_bounds(ComplexPlane(**plane_params))
        tile_plane = ComplexPlane(*tile_bounds, tile_spec.width, tile_spec.height)
        
        # Compute fractal for this tile
        result = fractal.compute(tile_plane, iterator)
        
        processing_time = time.time() - start_time
        
        return TileResult(
            tile_id=tile_spec.tile_id,
            iterations=result.iterations,
            escaped=result.escaped,
            final_values=result.final_values,
            x_start=tile_spec.x_start,
            y_start=tile_spec.y_start,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Error processing tile {tile_spec.tile_id if 'tile_spec' in locals() else '?'}: {e}")
        # Return empty result to avoid breaking the pipeline
        return TileResult(
            tile_id=-1,
            iterations=np.zeros((1, 1), dtype=np.int32),
            escaped=np.zeros((1, 1), dtype=bool),
            final_values=None,
            x_start=0,
            y_start=0,
            processing_time=0.0
        )


def assemble_tiles(tile_results: List[TileResult], total_width: int, total_height: int) -> IterationResult:
    """
    Assemble tile results into a complete fractal image.
    
    Args:
        tile_results: List of TileResult objects
        total_width: Total image width
        total_height: Total image height
        
    Returns:
        Complete IterationResult
    """
    # Initialize output arrays
    iterations = np.zeros((total_height, total_width), dtype=np.int32)
    escaped = np.zeros((total_height, total_width), dtype=bool)
    final_values = None
    
    # Check if any tile has final values
    has_final_values = any(tr.final_values is not None for tr in tile_results if tr.tile_id >= 0)
    if has_final_values:
        final_values = np.zeros((total_height, total_width), dtype=np.complex128)
    
    # Assemble tiles
    for tile_result in tile_results:
        if tile_result.tile_id < 0:  # Skip error tiles
            continue
        
        x_start = tile_result.x_start
        y_start = tile_result.y_start
        tile_height, tile_width = tile_result.iterations.shape
        
        # Place tile data in the correct position
        iterations[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_result.iterations
        escaped[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_result.escaped
        
        if has_final_values and tile_result.final_values is not None:
            final_values[y_start:y_start+tile_height, x_start:x_start+tile_width] = tile_result.final_values
    
    return IterationResult(iterations, escaped, final_values)


class MultiprocessingAccelerator:
    """Multiprocessing-based parallel fractal computation."""
    
    def __init__(self, num_processes: Optional[int] = None, tile_size: int = 256):
        """
        Initialize multiprocessing accelerator.
        
        Args:
            num_processes: Number of worker processes (None for CPU count)
            tile_size: Size of tiles for parallel processing
        """
        if num_processes is None:
            self.num_processes = mp.cpu_count()
        else:
            self.num_processes = max(1, num_processes)
        
        self.tile_size = tile_size
        logger.info(f"Multiprocessing accelerator: {self.num_processes} processes, {tile_size}x{tile_size} tiles")
    
    def render_fractal_parallel(self, fractal: FractalType, plane: ComplexPlane,
                               iterator: FractalIterator) -> IterationResult:
        """
        Render fractal using parallel tile-based processing.
        
        Args:
            fractal: Fractal type to render
            plane: Complex plane specification
            iterator: Fractal iterator
            
        Returns:
            Complete fractal computation result
        """
        start_time = time.time()
        
        # Create tile grid
        tiles = create_tile_grid(plane.width, plane.height, self.tile_size)
        
        # Prepare parameters for worker processes
        fractal_type = fractal.__class__.__name__.lower().replace('set', '').replace('fractal', '')
        if hasattr(fractal, 'parameters'):
            fractal_params = fractal.parameters.to_dict()
        else:
            fractal_params = {}
        
        iterator_params = {
            'max_iter': iterator.max_iter,
            'escape_radius': iterator.escape_radius,
            'dtype': iterator.dtype
        }
        
        plane_params = {
            'xmin': plane.xmin,
            'xmax': plane.xmax,
            'ymin': plane.ymin,
            'ymax': plane.ymax,
            'width': plane.width,
            'height': plane.height
        }
        
        # Prepare arguments for each tile
        tile_args = []
        for tile in tiles:
            args = (fractal_type, fractal_params, tile, iterator_params, plane_params)
            tile_args.append(args)
        
        # Process tiles in parallel
        logger.info(f"Processing {len(tiles)} tiles with {self.num_processes} processes")
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all tile jobs
            future_to_tile = {executor.submit(process_fractal_tile, args): i 
                             for i, args in enumerate(tile_args)}
            
            # Collect results as they complete
            tile_results = [None] * len(tiles)
            completed = 0
            
            for future in as_completed(future_to_tile):
                tile_idx = future_to_tile[future]
                try:
                    result = future.result()
                    tile_results[tile_idx] = result
                    completed += 1
                    
                    if completed % max(1, len(tiles) // 10) == 0:
                        progress = (completed / len(tiles)) * 100
                        logger.info(f"Completed {completed}/{len(tiles)} tiles ({progress:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Tile {tile_idx} failed: {e}")
                    # Create dummy result to avoid breaking assembly
                    tile_results[tile_idx] = TileResult(
                        tile_id=-1, iterations=np.zeros((1, 1), dtype=np.int32),
                        escaped=np.zeros((1, 1), dtype=bool), final_values=None,
                        x_start=0, y_start=0, processing_time=0.0
                    )
        
        # Assemble results
        logger.info("Assembling tile results")
        result = assemble_tiles(tile_results, plane.width, plane.height)
        
        total_time = time.time() - start_time
        total_processing_time = sum(tr.processing_time for tr in tile_results if tr.tile_id >= 0)
        
        logger.info(f"Parallel rendering complete: {total_time:.2f}s total, "
                   f"{total_processing_time:.2f}s processing time, "
                   f"efficiency: {total_processing_time/total_time:.2f}")
        
        return result
    
    def benchmark_parallel_performance(self, size=(2000, 2000), max_iter=1000) -> Dict[str, Any]:
        """
        Benchmark parallel vs sequential performance.
        
        Args:
            size: Image size for benchmark
            max_iter: Maximum iterations
            
        Returns:
            Performance comparison data
        """
        from ..core.fractal_types import MandelbrotSet
        
        width, height = size
        plane = ComplexPlane(-2.5, 1.0, -1.25, 1.25, width, height)
        fractal = MandelbrotSet()
        iterator = FractalIterator(max_iter, 2.0)
        
        results = {
            'resolution': f'{width}x{height}',
            'max_iterations': max_iter,
            'num_processes': self.num_processes,
            'tile_size': self.tile_size
        }
        
        # Benchmark parallel rendering
        start_time = time.time()
        parallel_result = self.render_fractal_parallel(fractal, plane, iterator)
        parallel_time = time.time() - start_time
        
        results['parallel_time'] = parallel_time
        results['parallel_pixels_per_second'] = (width * height) / parallel_time
        
        # Benchmark sequential rendering for comparison
        start_time = time.time()
        sequential_result = fractal.compute(plane, iterator)
        sequential_time = time.time() - start_time
        
        results['sequential_time'] = sequential_time
        results['sequential_pixels_per_second'] = (width * height) / sequential_time
        results['speedup'] = sequential_time / parallel_time
        results['efficiency'] = results['speedup'] / self.num_processes
        
        return results


class ProgressiveRenderer:
    """Progressive fractal renderer that shows results as tiles complete."""
    
    def __init__(self, num_processes: Optional[int] = None, tile_size: int = 256):
        """Initialize progressive renderer."""
        self.accelerator = MultiprocessingAccelerator(num_processes, tile_size)
        self.progress_callbacks = []
    
    def add_progress_callback(self, callback: Callable[[int, int, np.ndarray], None]):
        """
        Add callback for progress updates.
        
        Args:
            callback: Function called with (completed_tiles, total_tiles, current_image)
        """
        self.progress_callbacks.append(callback)
    
    def render_progressive(self, fractal: FractalType, plane: ComplexPlane,
                          iterator: FractalIterator) -> IterationResult:
        """
        Render fractal progressively, calling callbacks as tiles complete.
        
        Args:
            fractal: Fractal type to render
            plane: Complex plane specification
            iterator: Fractal iterator
            
        Returns:
            Complete fractal computation result
        """
        # Create tile grid
        tiles = create_tile_grid(plane.width, plane.height, self.accelerator.tile_size)
        
        # Initialize result arrays
        iterations = np.zeros((plane.height, plane.width), dtype=np.int32)
        escaped = np.zeros((plane.height, plane.width), dtype=bool)
        final_values = np.zeros((plane.height, plane.width), dtype=np.complex128)
        
        # Prepare parameters (same as MultiprocessingAccelerator)
        fractal_type = fractal.__class__.__name__.lower().replace('set', '').replace('fractal', '')
        fractal_params = fractal.parameters.to_dict() if hasattr(fractal, 'parameters') else {}
        
        iterator_params = {
            'max_iter': iterator.max_iter,
            'escape_radius': iterator.escape_radius,
            'dtype': iterator.dtype
        }
        
        plane_params = {
            'xmin': plane.xmin, 'xmax': plane.xmax,
            'ymin': plane.ymin, 'ymax': plane.ymax,
            'width': plane.width, 'height': plane.height
        }
        
        tile_args = [(fractal_type, fractal_params, tile, iterator_params, plane_params) 
                     for tile in tiles]
        
        # Process with progress updates
        with ProcessPoolExecutor(max_workers=self.accelerator.num_processes) as executor:
            future_to_tile = {executor.submit(process_fractal_tile, args): i 
                             for i, args in enumerate(tile_args)}
            
            completed = 0
            has_final_values = False
            
            for future in as_completed(future_to_tile):
                tile_idx = future_to_tile[future]
                
                try:
                    result = future.result()
                    if result.tile_id >= 0:
                        # Update result arrays
                        x_start, y_start = result.x_start, result.y_start
                        tile_h, tile_w = result.iterations.shape
                        
                        iterations[y_start:y_start+tile_h, x_start:x_start+tile_w] = result.iterations
                        escaped[y_start:y_start+tile_h, x_start:x_start+tile_w] = result.escaped
                        
                        if result.final_values is not None:
                            has_final_values = True
                            final_values[y_start:y_start+tile_h, x_start:x_start+tile_w] = result.final_values
                    
                    completed += 1
                    
                    # Call progress callbacks
                    for callback in self.progress_callbacks:
                        try:
                            callback(completed, len(tiles), iterations.copy())
                        except Exception as e:
                            logger.warning(f"Progress callback failed: {e}")
                
                except Exception as e:
                    logger.error(f"Tile processing failed: {e}")
                    completed += 1
        
        # Return final result
        final_vals = final_values if has_final_values else None
        return IterationResult(iterations, escaped, final_vals)


class TileCache:
    """Cache system for computed fractal tiles."""
    
    def __init__(self, max_size_mb: int = 100):
        """
        Initialize tile cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.current_size = 0
    
    def _get_tile_key(self, fractal_type: str, fractal_params: dict,
                     bounds: Tuple[float, float, float, float],
                     max_iter: int, escape_radius: float) -> str:
        """Generate cache key for tile parameters."""
        import hashlib
        
        key_data = f"{fractal_type}_{fractal_params}_{bounds}_{max_iter}_{escape_radius}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[TileResult]:
        """Get cached tile result."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, result: TileResult):
        """Cache tile result."""
        # Estimate memory usage
        size = result.iterations.nbytes + result.escaped.nbytes
        if result.final_values is not None:
            size += result.final_values.nbytes
        
        # Evict old entries if needed
        while self.current_size + size > self.max_size_bytes and self.cache:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove(oldest_key)
        
        # Add new entry
        self.cache[key] = result
        self.access_times[key] = time.time()
        self.current_size += size
    
    def _remove(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            result = self.cache[key]
            size = result.iterations.nbytes + result.escaped.nbytes
            if result.final_values is not None:
                size += result.final_values.nbytes
            
            del self.cache[key]
            del self.access_times[key]
            self.current_size -= size
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()
        self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self.cache),
            'size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'hit_rate': getattr(self, '_hits', 0) / max(1, getattr(self, '_requests', 1))
        }


class AdaptiveTileSize:
    """Automatically adjust tile sizes based on computation complexity."""
    
    def __init__(self, base_tile_size: int = 256, min_tile_size: int = 64, max_tile_size: int = 1024):
        """Initialize adaptive tile sizing."""
        self.base_tile_size = base_tile_size
        self.min_tile_size = min_tile_size
        self.max_tile_size = max_tile_size
        self.performance_history = []
    
    def get_optimal_tile_size(self, plane: ComplexPlane, max_iter: int) -> int:
        """
        Calculate optimal tile size based on problem characteristics.
        
        Args:
            plane: Complex plane specification
            max_iter: Maximum iterations
            
        Returns:
            Recommended tile size
        """
        # Estimate complexity based on zoom level and iterations
        zoom_level = 1.0 / ((plane.xmax - plane.xmin) * (plane.ymax - plane.ymin))
        complexity_factor = np.log10(max_iter) * np.log10(max(1, zoom_level))
        
        # Adjust tile size based on complexity
        if complexity_factor > 10:  # High complexity
            tile_size = max(self.min_tile_size, self.base_tile_size // 2)
        elif complexity_factor < 5:  # Low complexity
            tile_size = min(self.max_tile_size, self.base_tile_size * 2)
        else:
            tile_size = self.base_tile_size
        
        # Consider available memory
        total_pixels = plane.width * plane.height
        available_memory_mb = self._estimate_available_memory()
        memory_per_pixel = 32  # bytes (rough estimate including all arrays)
        
        max_tile_pixels = (available_memory_mb * 1024 * 1024) // (memory_per_pixel * mp.cpu_count())
        max_tile_size_memory = int(np.sqrt(max_tile_pixels))
        
        tile_size = min(tile_size, max_tile_size_memory, self.max_tile_size)
        tile_size = max(tile_size, self.min_tile_size)
        
        logger.info(f"Adaptive tile size: {tile_size}x{tile_size} "
                   f"(complexity: {complexity_factor:.1f}, memory limit: {max_tile_size_memory})")
        
        return tile_size
    
    def _estimate_available_memory(self) -> int:
        """Estimate available memory in MB."""
        try:
            import psutil
            return int(psutil.virtual_memory().available / (1024 * 1024))
        except ImportError:
            # Rough estimate if psutil not available
            return 4000  # 4GB default assumption


# Global multiprocessing accelerator
_mp_accelerator = None


def get_multiprocessing_accelerator(num_processes=None, tile_size=256):
    """Get the global multiprocessing accelerator instance."""
    global _mp_accelerator
    if _mp_accelerator is None or _mp_accelerator.num_processes != (num_processes or mp.cpu_count()):
        _mp_accelerator = MultiprocessingAccelerator(num_processes, tile_size)
    return _mp_accelerator


def get_optimal_process_count():
    """Get optimal number of processes for fractal computation."""
    cpu_count = mp.cpu_count()
    
    # Leave one core for system
    optimal = max(1, cpu_count - 1)
    
    # Consider memory constraints
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        # Rough estimate: 1 process per 2GB available memory
        memory_limited = max(1, int(available_gb / 2))
        optimal = min(optimal, memory_limited)
    except ImportError:
        pass
    
    return optimal
