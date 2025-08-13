"""
Command-line interface for fractal generation.

This module provides a comprehensive CLI for generating fractals with
all available features and configuration options.
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import time

from .. import __version__
from ..api import FractalRenderer, RenderConfig, BatchRenderer, FractalExplorer
from ..core.fractal_types import FractalRegistry, JULIA_PRESETS
from ..io.config import ConfigManager, load_config_from_args, EnvironmentConfig
from ..tools.animation import AnimationSequencer
from ..accelerators.numba_backend import is_numba_available
from ..accelerators.gpu_backend import is_gpu_available, compare_gpu_cpu_performance

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
@click.option('--preset', help='Configuration preset to use')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress most output')
@click.pass_context
def main(ctx, version, config, preset, verbose, quiet):
    """
    Fractal Generator - High-quality fractal image generation tool.
    
    Generate beautiful fractal images with advanced rendering capabilities
    including GPU acceleration, arbitrary precision, and multiple coloring algorithms.
    """
    # Setup logging
    if quiet:
        logging.basicConfig(level=logging.ERROR)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG, 
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                          format='%(levelname)s: %(message)s')
    
    if version:
        click.echo(f"Fractal Generator v{__version__}")
        click.echo(f"Python: {sys.version}")
        
        # Show acceleration support
        click.echo(f"Numba acceleration: {'Available' if is_numba_available() else 'Not available'}")
        click.echo(f"GPU acceleration: {'Available' if is_gpu_available() else 'Not available'}")
        
        if ctx.invoked_subcommand is None:
            sys.exit(0)
    
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config
    ctx.obj['preset'] = preset
    ctx.obj['verbose'] = verbose


@main.command()
@click.argument('fractal_type', type=click.Choice(['mandelbrot', 'julia', 'burning_ship', 'multibrot']))
@click.argument('output', type=click.Path())
@click.option('--width', '-w', type=int, help='Image width')
@click.option('--height', '-h', type=int, help='Image height')
@click.option('--bounds', type=str, help='Complex plane bounds: "xmin,xmax,ymin,ymax"')
@click.option('--max-iter', type=int, help='Maximum iterations')
@click.option('--escape-radius', type=float, help='Escape radius')
@click.option('--precision', type=click.Choice(['single', 'double', 'quad']), help='Numerical precision')
@click.option('--palette', help='Color palette name')
@click.option('--algorithm', type=click.Choice(['escape_time', 'smooth', 'histogram']), 
              help='Coloring algorithm')
@click.option('--julia-c', type=str, help='Julia constant (real,imag) or preset name')
@click.option('--multibrot-power', type=float, help='Multibrot power')
@click.option('--no-gpu', is_flag=True, help='Disable GPU acceleration')
@click.option('--no-numba', is_flag=True, help='Disable Numba acceleration')
@click.option('--processes', type=int, help='Number of processes for parallel rendering')
@click.option('--tile-size', type=int, help='Tile size for parallel rendering')
@click.option('--antialiasing', type=int, help='Antialiasing factor')
@click.pass_context
def render(ctx, fractal_type, output, **kwargs):
    """
    Render a single fractal image.
    
    FRACTAL_TYPE: Type of fractal (mandelbrot, julia, burning_ship, multibrot)
    OUTPUT: Output image file path
    """
    try:
        # Load configuration
        render_config, fractal_configs = load_config_from_args(
            ctx.obj.get('config_file'), 
            ctx.obj.get('preset')
        )
        
        # Apply command-line overrides
        overrides = {k.replace('_', ''): v for k, v in kwargs.items() 
                    if v is not None and k not in ['julia_c', 'multibrot_power']}
        
        # Handle special parameters
        if kwargs.get('bounds'):
            try:
                bounds = [float(x.strip()) for x in kwargs['bounds'].split(',')]
                if len(bounds) == 4:
                    overrides['bounds'] = tuple(bounds)
            except ValueError:
                click.echo("Error: Invalid bounds format. Use 'xmin,xmax,ymin,ymax'", err=True)
                sys.exit(1)
        
        # Apply boolean flags
        if kwargs.get('no_gpu'):
            overrides['use_gpu'] = False
        if kwargs.get('no_numba'):
            overrides['use_numba'] = False
        
        # Update config
        for key, value in overrides.items():
            if hasattr(render_config, key):
                setattr(render_config, key, value)
        
        # Create fractal
        fractal_params = fractal_configs.get(fractal_type, {})
        
        # Handle Julia constant
        if fractal_type == 'julia' and kwargs.get('julia_c'):
            julia_c = kwargs['julia_c']
            if julia_c in JULIA_PRESETS:
                fractal_params.update(JULIA_PRESETS[julia_c].to_dict())
                click.echo(f"Using Julia preset: {julia_c}")
            else:
                try:
                    parts = [float(x.strip()) for x in julia_c.split(',')]
                    if len(parts) == 2:
                        fractal_params['c_real'] = parts[0]
                        fractal_params['c_imag'] = parts[1]
                except ValueError:
                    click.echo("Error: Invalid Julia constant. Use 'real,imag' or preset name", err=True)
                    sys.exit(1)
        
        # Handle Multibrot power
        if fractal_type == 'multibrot' and kwargs.get('multibrot_power'):
            fractal_params['power'] = kwargs['multibrot_power']
        
        fractal = FractalRegistry.create_fractal(fractal_type, **fractal_params)
        
        # Create renderer
        renderer = FractalRenderer(render_config)
        
        # Progress callback
        def progress_callback(progress, image=None):
            if ctx.obj.get('verbose'):
                click.echo(f"Progress: {progress*100:.1f}%")
        
        # Render
        click.echo(f"Rendering {fractal_type} fractal...")
        start_time = time.time()
        
        image = renderer.render(fractal, Path(output), progress_callback)
        
        render_time = time.time() - start_time
        click.echo(f"Render complete: {render_time:.2f}s")
        click.echo(f"Saved: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='batch_output',
              help='Output directory for batch renders')
@click.option('--dry-run', is_flag=True, help='Show what would be rendered without actually rendering')
@click.pass_context
def batch(ctx, config_file, output_dir, dry_run):
    """
    Execute batch rendering jobs from configuration file.
    
    CONFIG_FILE: Batch configuration file (JSON or YAML)
    """
    try:
        # Load batch configuration
        manager = ConfigManager()
        batch_config = manager.load_config(config_file)
        
        if 'batch_jobs' not in batch_config:
            click.echo("Error: No 'batch_jobs' section found in config file", err=True)
            sys.exit(1)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create batch renderer
        base_config = manager.create_render_config(batch_config)
        batch_renderer = BatchRenderer(base_config)
        
        # Add jobs from configuration
        for job_config in batch_config['batch_jobs']:
            fractal_type = job_config['fractal_type']
            fractal_params = job_config.get('fractal_params', {})
            render_overrides = job_config.get('render_config', {})
            
            # Create fractal
            fractal = FractalRegistry.create_fractal(fractal_type, **fractal_params)
            
            # Generate output filename
            job_name = job_config.get('name', f"{fractal_type}_{len(batch_renderer.jobs)}")
            output_file = output_path / f"{job_name}.png"
            
            if dry_run:
                click.echo(f"Would render: {job_name} -> {output_file}")
                continue
            
            batch_renderer.add_job(fractal, output_file, render_overrides, job_name)
        
        if dry_run:
            click.echo(f"Dry run complete. {len(batch_config['batch_jobs'])} jobs would be executed.")
            return
        
        # Execute batch
        click.echo(f"Starting batch render: {len(batch_renderer.jobs)} jobs")
        
        def progress_callback(completed, total, result):
            click.echo(f"Completed {completed}/{total}: {result['job_name']}")
        
        results = batch_renderer.run_batch(progress_callback)
        
        # Show summary
        summary = batch_renderer.get_summary()
        click.echo(f"\nBatch complete:")
        click.echo(f"  Jobs completed: {summary['completed']}/{summary['total_jobs']}")
        click.echo(f"  Success rate: {summary['success_rate']*100:.1f}%")
        click.echo(f"  Total time: {summary['total_render_time']:.2f}s")
        click.echo(f"  Average time: {summary['average_render_time']:.2f}s per job")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('fractal_type', type=click.Choice(['mandelbrot', 'julia']))
@click.argument('output_dir', type=click.Path())
@click.option('--frames', type=int, default=60, help='Number of animation frames')
@click.option('--zoom-factor', type=float, default=1.5, help='Zoom factor per frame')
@click.option('--center', type=str, help='Zoom center point "real,imag"')
@click.option('--fps', type=int, default=30, help='Frames per second for video')
@click.option('--format', type=click.Choice(['images', 'mp4', 'gif']), default='images',
              help='Output format')
@click.pass_context
def animate(ctx, fractal_type, output_dir, frames, zoom_factor, center, fps, format):
    """
    Generate fractal animation sequences.
    
    FRACTAL_TYPE: Type of fractal to animate
    OUTPUT_DIR: Output directory for animation frames/video
    """
    try:
        # Load configuration
        render_config, fractal_configs = load_config_from_args(
            ctx.obj.get('config_file'), 
            ctx.obj.get('preset')
        )
        
        # Parse center point
        if center:
            try:
                center_parts = [float(x.strip()) for x in center.split(',')]
                if len(center_parts) != 2:
                    raise ValueError("Center must have exactly 2 coordinates")
                center_point = complex(center_parts[0], center_parts[1])
            except ValueError:
                click.echo("Error: Invalid center format. Use 'real,imag'", err=True)
                sys.exit(1)
        else:
            # Use fractal's recommended center
            fractal = FractalRegistry.create_fractal(fractal_type, **fractal_configs.get(fractal_type, {}))
            bounds = fractal.get_recommended_bounds()
            center_point = complex((bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2)
        
        # Create animation sequencer
        sequencer = AnimationSequencer(render_config)
        
        click.echo(f"Creating {frames} frame zoom animation...")
        click.echo(f"Center: {center_point}")
        click.echo(f"Zoom factor: {zoom_factor} per frame")
        
        # Generate zoom sequence
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        def progress_callback(frame_num, total_frames, current_image=None):
            click.echo(f"Rendering frame {frame_num}/{total_frames}")
        
        # Create fractal for animation
        fractal = FractalRegistry.create_fractal(fractal_type, **fractal_configs.get(fractal_type, {}))
        
        if format == 'images':
            sequencer.create_zoom_sequence(
                fractal, center_point, zoom_factor, frames,
                output_path, progress_callback
            )
            click.echo(f"Animation frames saved to: {output_path}")
        
        elif format in ['mp4', 'gif']:
            # First create image sequence
            temp_dir = output_path / 'temp_frames'
            temp_dir.mkdir(exist_ok=True)
            
            sequencer.create_zoom_sequence(
                fractal, center_point, zoom_factor, frames,
                temp_dir, progress_callback
            )
            
            # Then create video/gif
            click.echo(f"Creating {format.upper()} animation...")
            
            if format == 'mp4':
                video_path = output_path / 'animation.mp4'
                sequencer.create_video_from_images(temp_dir, video_path, fps)
                click.echo(f"Video saved to: {video_path}")
            
            elif format == 'gif':
                gif_path = output_path / 'animation.gif'
                sequencer.create_gif_from_images(temp_dir, gif_path, fps//3)  # Slower for GIF
                click.echo(f"GIF saved to: {gif_path}")
            
            # Clean up temp frames
            import shutil
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--size', type=str, default='1000x1000', help='Benchmark image size (widthxheight)')
@click.option('--iterations', type=int, default=1000, help='Maximum iterations for benchmark')
@click.option('--compare-all', is_flag=True, help='Compare all available acceleration methods')
@click.pass_context
def benchmark(ctx, size, iterations, compare_all):
    """
    Benchmark rendering performance with different acceleration methods.
    """
    try:
        # Parse size
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            click.echo("Error: Invalid size format. Use 'widthxheight'", err=True)
            sys.exit(1)
        
        click.echo(f"Fractal Generator Performance Benchmark")
        click.echo(f"Image size: {width}x{height} ({width*height:,} pixels)")
        click.echo(f"Max iterations: {iterations}")
        click.echo("")
        
        if compare_all:
            # Comprehensive benchmark
            results = compare_gpu_cpu_performance((width, height), iterations)
            
            click.echo("Performance Results:")
            click.echo(f"  CPU (NumPy): {results['cpu_time']:.2f}s "
                      f"({results['cpu_pixels_per_second']:,.0f} pixels/sec)")
            
            if 'gpu_time' in results:
                click.echo(f"  GPU ({results['gpu_backend']}): {results['gpu_time']:.2f}s "
                          f"({results['gpu_pixels_per_second']:,.0f} pixels/sec)")
                click.echo(f"  GPU Speedup: {results['speedup']:.2f}x")
            else:
                click.echo(f"  GPU: {results.get('gpu_error', 'Not available')}")
        
        else:
            # Test current configuration
            config = RenderConfig(width=width, height=height, max_iterations=iterations)
            renderer = FractalRenderer(config)
            
            benchmark_results = renderer.benchmark_performance()
            
            click.echo("Configuration:")
            for key, value in benchmark_results['config'].items():
                click.echo(f"  {key}: {value}")
            
            click.echo("\nPerformance Results:")
            for method, result in benchmark_results['benchmarks'].items():
                if 'error' in result:
                    click.echo(f"  {method.upper()}: {result['error']}")
                else:
                    time_val = result.get('time', result.get('gpu_time', result.get('parallel_time', 0)))
                    pps = result.get('pixels_per_second', 0)
                    click.echo(f"  {method.upper()}: {time_val:.2f}s ({pps:,.0f} pixels/sec)")
                    
                    if 'speedup' in result:
                        click.echo(f"    Speedup: {result['speedup']:.2f}x")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.option('--output', '-o', type=click.Path(), default='config_template.yaml',
              help='Output file path')
@click.option('--format', type=click.Choice(['yaml', 'json']), help='Output format (auto-detect if not specified)')
@click.option('--with-examples', is_flag=True, help='Include example values')
@click.pass_context
def init_config(ctx, output, format, with_examples):
    """
    Create a configuration template file.
    """
    try:
        output_path = Path(output)
        
        # Auto-detect format if not specified
        if not format:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                format = 'yaml'
            elif output_path.suffix.lower() == '.json':
                format = 'json'
            else:
                format = 'yaml'  # Default
                if not output_path.suffix:
                    output_path = output_path.with_suffix('.yaml')
        
        manager = ConfigManager()
        manager.export_config_template(output_path, with_examples)
        
        click.echo(f"Configuration template created: {output_path}")
        click.echo(f"Format: {format.upper()}")
        
        if with_examples:
            click.echo("Template includes example values - edit as needed.")
        else:
            click.echo("Template includes documentation comments - fill in values.")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.pass_context
def list_presets(ctx):
    """List available configuration presets."""
    try:
        manager = ConfigManager()
        config_dict = manager.load_config()
        presets = manager.list_presets(config_dict)
        
        if not presets:
            click.echo("No presets available.")
            return
        
        click.echo("Available presets:")
        for preset in presets:
            click.echo(f"  {preset}")
            
            # Show preset details if verbose
            if ctx.obj.get('verbose'):
                preset_config = config_dict['presets'][preset]
                if '_description' in preset_config:
                    click.echo(f"    Description: {preset_config['_description']}")
                
                key_settings = {k: v for k, v in preset_config.items() 
                              if not k.startswith('_')}
                for key, value in key_settings.items():
                    click.echo(f"    {key}: {value}")
                click.echo()
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def list_fractals(ctx):
    """List available fractal types and their parameters."""
    try:
        fractals = FractalRegistry.list_fractals()
        
        click.echo("Available fractal types:")
        for name, description in fractals.items():
            click.echo(f"  {name}")
            if ctx.obj.get('verbose'):
                click.echo(f"    {description}")
        
        click.echo(f"\nJulia set presets:")
        for name, params in JULIA_PRESETS.items():
            click.echo(f"  {name}: c = {params.c}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def list_palettes(ctx):
    """List available color palettes."""
    try:
        from ..rendering.coloring import ColoringEngine
        
        engine = ColoringEngine()
        palettes = engine.list_palettes()
        algorithms = engine.list_algorithms()
        
        click.echo("Available color palettes:")
        for palette in palettes:
            click.echo(f"  {palette}")
        
        click.echo(f"\nAvailable coloring algorithms:")
        for algorithm in algorithms:
            click.echo(f"  {algorithm}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def system_info(ctx):
    """Display system capabilities and configuration recommendations."""
    try:
        capabilities = EnvironmentConfig.detect_system_capabilities()
        
        click.echo("System Information:")
        click.echo(f"  CPU cores: {capabilities['cpu_count']}")
        click.echo(f"  Memory: {capabilities['memory_gb']:.1f} GB")
        click.echo(f"  Numba: {'Available' if capabilities['numba_available'] else 'Not available'}")
        
        if capabilities['numba_available']:
            click.echo(f"    Version: {capabilities.get('numba_version', 'Unknown')}")
        
        click.echo(f"  GPU: {'Available' if capabilities['gpu_available'] else 'Not available'}")
        
        if capabilities['gpu_available']:
            click.echo(f"    Backend: {capabilities['gpu_backend']}")
            if 'gpu_memory_gb' in capabilities:
                click.echo(f"    Memory: {capabilities['gpu_memory_gb']:.1f} GB")
        
        # Show recommendations
        click.echo("\nRecommended configurations:")
        
        for use_case in ['preview', 'general', 'high_quality', 'research']:
            try:
                rec_config = EnvironmentConfig.recommend_config(use_case)['render']
                click.echo(f"  {use_case.title()}:")
                
                key_params = ['width', 'height', 'max_iterations', 'use_gpu', 'use_multiprocessing']
                for param in key_params:
                    if param in rec_config:
                        click.echo(f"    {param}: {rec_config[param]}")
            
            except Exception as e:
                if ctx.obj.get('verbose'):
                    click.echo(f"    Error generating {use_case} config: {e}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def validate_config(ctx, config_file):
    """Validate a configuration file."""
    try:
        manager = ConfigManager()
        config_dict = manager.load_config(config_file)
        
        # Validate configuration
        errors = manager.validate_config(config_dict)
        
        if not errors:
            click.echo(f"✓ Configuration file is valid: {config_file}")
            
            # Show performance warnings if verbose
            if ctx.obj.get('verbose'):
                from ..io.config import ConfigValidator
                render_config = manager.create_render_config(config_dict)
                warnings = ConfigValidator.validate_performance_config(render_config)
                
                if warnings:
                    click.echo("\nPerformance warnings:")
                    for warning in warnings:
                        click.echo(f"  ⚠ {warning}")
        else:
            click.echo(f"✗ Configuration file has errors: {config_file}")
            for error in errors:
                click.echo(f"  Error: {error}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error validating config: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('fractal_type', type=click.Choice(['mandelbrot', 'julia']))
@click.option('--width', '-w', type=int, default=800, help='Window width')
@click.option('--height', '-h', type=int, default=600, help='Window height')
@click.option('--julia-c', type=str, help='Julia constant for Julia sets')
@click.pass_context
def explore(ctx, fractal_type, width, height, julia_c):
    """
    Interactive fractal exploration (requires GUI dependencies).
    
    FRACTAL_TYPE: Type of fractal to explore
    """
    try:
        # Check if GUI dependencies are available
        try:
            import tkinter as tk
            from tkinter import ttk
            from PIL import Image, ImageTk
        except ImportError as e:
            click.echo(f"Error: GUI dependencies not available: {e}", err=True)
            click.echo("Install with: pip install tkinter pillow", err=True)
            sys.exit(1)
        
        # Load configuration
        render_config, fractal_configs = load_config_from_args(
            ctx.obj.get('config_file'), 
            ctx.obj.get('preset')
        )
        
        # Override size
        render_config.width = width
        render_config.height = height
        
        # Create explorer
        explorer = FractalExplorer(render_config)
        
        # Set up fractal
        fractal_params = fractal_configs.get(fractal_type, {})
        
        if fractal_type == 'julia' and julia_c:
            if julia_c in JULIA_PRESETS:
                fractal_params.update(JULIA_PRESETS[julia_c].to_dict())
            else:
                try:
                    parts = [float(x.strip()) for x in julia_c.split(',')]
                    if len(parts) == 2:
                        fractal_params['c_real'] = parts[0]
                        fractal_params['c_imag'] = parts[1]
                except ValueError:
                    click.echo("Error: Invalid Julia constant format", err=True)
                    sys.exit(1)
        
        fractal = FractalRegistry.create_fractal(fractal_type, **fractal_params)
        explorer.set_fractal(fractal)
        
        # Create simple GUI
        root = tk.Tk()
        root.title(f"Fractal Explorer - {fractal_type.title()}")
        
        # Canvas for fractal display
        canvas = tk.Canvas(root, width=width, height=height, bg='black')
        canvas.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Control panel
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Current image reference
        current_photo = None
        
        def update_display():
            nonlocal current_photo
            try:
                # Render current view
                rgb_image = explorer.render_current()
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray((rgb_image * 255).astype('uint8'))
                current_photo = ImageTk.PhotoImage(pil_image)
                
                # Update canvas
                canvas.delete("all")
                canvas.create_image(width//2, height//2, image=current_photo)
                
                # Update status
                info = explorer.get_exploration_info()
                status_var.set(f"Zoom: {info['zoom_level']:.2e} | Iterations: {info['max_iterations']}")
                
            except Exception as e:
                click.echo(f"Render error: {e}")
        
        def on_canvas_click(event):
            try:
                explorer.zoom_to_point(event.x, event.y, 2.0)
                update_display()
            except Exception as e:
                click.echo(f"Zoom error: {e}")
        
        def zoom_out():
            try:
                explorer.zoom_out(0.5)
                update_display()
            except Exception as e:
                click.echo(f"Zoom out error: {e}")
        
        def go_back():
            try:
                explorer.go_back()
                update_display()
            except Exception as e:
                click.echo(f"Go back error: {e}")
        
        def reset_view():
            try:
                explorer.reset_view()
                update_display()
            except Exception as e:
                click.echo(f"Reset error: {e}")
        
        def save_image():
            try:
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
                )
                if filename:
                    rgb_image = explorer.render_current()
                    Image.fromarray((rgb_image * 255).astype('uint8')).save(filename)
                    click.echo(f"Saved: {filename}")
            except Exception as e:
                click.echo(f"Save error: {e}")
        
        # Bind canvas click
        canvas.bind("<Button-1>", on_canvas_click)
        
        # Control buttons
        ttk.Button(control_frame, text="Zoom Out", command=zoom_out).pack(pady=2)
        ttk.Button(control_frame, text="Go Back", command=go_back).pack(pady=2)
        ttk.Button(control_frame, text="Reset View", command=reset_view).pack(pady=2)
        ttk.Button(control_frame, text="Save Image", command=save_image).pack(pady=2)
        
        # Status display
        status_var = tk.StringVar()
        status_label = ttk.Label(control_frame, textvariable=status_var, wraplength=200)
        status_label.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(control_frame, 
                                text="Click to zoom in\nUse buttons to navigate",
                                wraplength=200)
        instructions.pack(pady=5)
        
        # Initial render
        click.echo("Rendering initial view...")
        update_display()
        
        click.echo(f"Starting interactive explorer...")
        click.echo(f"Click on the image to zoom in")
        
        # Start GUI
        root.mainloop()
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
