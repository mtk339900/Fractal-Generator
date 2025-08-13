# Fractal Generator

A high-performance, production-ready Python library for generating beautiful fractal images with advanced rendering capabilities, GPU acceleration, and extensive customization options.

## Features

### Core Functionality
- **Multiple Fractal Types**: Mandelbrot sets, Julia sets, Burning Ship, Multibrot, and extensible custom fractals
- **Advanced Coloring**: Escape-time, smooth/continuous, histogram, and orbit trap coloring algorithms
- **Custom Palettes**: Built-in color palettes plus support for custom palettes and matplotlib colormap integration
- **High-Quality Output**: PNG, TIFF, JPEG export with embedded metadata and raw data support

### Performance & Scalability  
- **GPU Acceleration**: CUDA support via CuPy with automatic CPU fallback
- **JIT Compilation**: Numba acceleration for significant CPU performance improvements
- **Parallel Processing**: Multi-core tile-based rendering with intelligent load balancing
- **Memory Efficient**: Streaming tile rendering for ultra-high resolution images
- **Arbitrary Precision**: Deep zoom support using mpmath for extreme magnifications

### Advanced Features
- **Interactive Exploration**: GUI-based zoom and pan with real-time rendering
- **Animation Support**: Smooth zoom sequences and parameter animations with video export
- **Batch Processing**: Queue-based batch rendering with progress tracking  
- **Anti-aliasing**: Supersampling support for improved visual quality
- **Configuration System**: YAML/JSON configuration with presets and environment variable support

## Installation

### Basic Installation
```bash
pip install fractal-generator
```

### With GPU Support
```bash
pip install fractal-generator[gpu]
```

### With All Features
```bash
pip install fractal-generator[gpu,web,video,precision]
```

### Development Installation
```bash
git clone https://github.com/mtk339900/fractal-generator.git
cd fractal-generator
pip install -e .[dev]
```

## Quick Start

### Python API

```python
from fractal_generator import FractalRenderer, MandelbrotSet, RenderConfig

# Create a high-quality Mandelbrot set
config = RenderConfig(
    width=1920,
    height=1080,
    max_iterations=1000,
    coloring_algorithm='smooth',
    color_palette='hot',
    use_gpu=True
)

renderer = FractalRenderer(config)
fractal = MandelbrotSet()

# Render to numpy array
image = renderer.render(fractal)

# Or render directly to file
image = renderer.render(fractal, output_path='mandelbrot.png')
```

### Command Line Interface

```bash
# Render a basic Mandelbrot set
fractal-gen render mandelbrot output.png --width 1920 --height 1080

# High-quality Julia set with custom parameters
fractal-gen render julia julia.png --julia-c -0.75,0.1 --max-iter 2000 --palette viridis

# Create zoom animation
fractal-gen animate mandelbrot animation/ --frames 60 --zoom-factor 2.0 --center -0.5,0.0

# Batch processing
fractal-gen batch batch_config.yaml --output-dir renders/

# Interactive exploration
fractal-gen explore mandelbrot --width 800 --height 600
```

## Configuration

### YAML Configuration Example

```yaml
render:
  width: 1920
  height: 1080
  bounds: [-2.5, 1.0, -1.25, 1.25]
  max_iterations: 1000
  precision: double
  coloring_algorithm: smooth
  color_palette: hot
  use_gpu: true
  use_multiprocessing: true

fractals:
  mandelbrot:
    z0_real: 0.0
    z0_imag: 0.0
  julia:
    c_real: -0.75
    c_imag: 0.1

presets:
  high_quality:
    width: 3840
    height: 2160
    max_iterations: 2000
    antialiasing: 2
    output_format: png
```

### Environment Variables

```bash
export FRACTAL_WIDTH=1920
export FRACTAL_HEIGHT=1080
export FRACTAL_USE_GPU=true
export FRACTAL_MAX_ITER=2000
```

## Advanced Usage

### Custom Fractals

```python
from fractal_generator.core.fractal_types import CustomFractal

def tricorn_iteration(z):
    """Tricorn fractal: z = conj(z)^2 + c"""
    return np.conj(z)**2 + c

tricorn = CustomFractal(
    tricorn_iteration,
    parameters=CustomFractalParameters(
        description="Tricorn fractal",
        recommended_bounds=(-2, 2, -2, 2)
    )
)

image = renderer.render(tricorn)
```

### Arbitrary Precision Deep Zooms

```python
config = RenderConfig(
    precision='quad',  # or specific decimal places: 50
    max_iterations=5000,
    bounds=('1.25066', '1.25067', '0.02012', '0.02013')  # String for exact precision
)

renderer = FractalRenderer(config)
deep_zoom = renderer.render(fractal, 'deep_zoom.png')
```

### Custom Color Palettes

```python
from fractal_generator.rendering.coloring import Palette, ColorRGB

# Create custom palette
custom_palette = Palette([
    ColorRGB(0, 0, 0),      # Black
    ColorRGB(0.5, 0, 0.5),  # Purple
    ColorRGB(1, 0, 1),      # Magenta
    ColorRGB(1, 1, 1),      # White
], name="Custom Purple")

# Add to coloring engine
renderer.coloring_engine.add_palette("purple", custom_palette)
```

### Batch Processing

```python
from fractal_generator import BatchRenderer

batch = BatchRenderer()

# Add multiple jobs
batch.add_job(MandelbrotSet(), 'mandelbrot_1.png', 
              config_overrides={'max_iterations': 500})
batch.add_job(JuliaSet(), 'julia_1.png', 
              config_overrides={'color_palette': 'viridis'})

# Execute all jobs
results = batch.run_batch()
summary = batch.get_summary()
```

## Performance Optimization

### GPU Acceleration

The library automatically detects and uses GPU acceleration when available:

- **CUDA Support**: Via CuPy for maximum performance
- **Automatic Fallback**: Gracefully falls back to CPU if GPU unavailable
- **Memory Management**: Intelligent GPU memory allocation and cleanup

### Parallel Processing

- **Multi-core CPU**: Automatic tile-based parallel rendering
- **Optimal Threading**: Automatically detects optimal number of processes
- **Memory Efficient**: Streaming processing for large images

### Performance Tuning

```python
# Benchmark different methods
results = renderer.benchmark_performance()

# System capabilities detection
from fractal_generator.io.config import EnvironmentConfig
capabilities = EnvironmentConfig.detect_system_capabilities()
recommended = EnvironmentConfig.recommend_config('high_quality')
```

## Animation and Video

### Zoom Animations

```python
from fractal_generator.tools.animation import AnimationSequencer

sequencer = AnimationSequencer(config)

# Create zoom sequence
sequencer.create_zoom_sequence(
    fractal=MandelbrotSet(),
    center=complex(-0.5, 0.0),
    zoom_factor=1.5,
    frames=60,
    output_dir='zoom_animation/'
)

# Convert to video
sequencer.create_video_from_images('zoom_animation/', 'zoom.mp4', fps=30)
```

### Parameter Animations

```python
# Animate Julia set parameter
julia_animation = sequencer.create_parameter_animation(
    fractal_type='julia',
    parameter='c_real',
    values=np.linspace(-1.0, 0.0, 60),
    output_dir='julia_morph/'
)
```

## Interactive Exploration

### GUI Explorer

```python
from fractal_generator import FractalExplorer

explorer = FractalExplorer()
explorer.set_fractal(MandelbrotSet())

# Zoom to specific point
explorer.zoom_to_point(400, 300, zoom_factor=2.0)

# Pan the view
explorer.pan(0.1, 0.1)

# Go back to previous view
explorer.go_back()

# Change parameters
explorer.adjust_iterations(2000)
explorer.change_palette('viridis')

# Get current state
info = explorer.get_exploration_info()
```

### Web Interface (Optional)

```python
# Requires streamlit: pip install fractal-generator[web]
import streamlit as st
from fractal_generator.tools.interactive import create_streamlit_app

create_streamlit_app()
```

## Command Line Reference

### Basic Commands

```bash
# Render fractals
fractal-gen render <fractal_type> <output> [options]

# Batch processing  
fractal-gen batch <config_file> [options]

# Animation generation
fractal-gen animate <fractal_type> <output_dir> [options]

# Interactive exploration
fractal-gen explore <fractal_type> [options]

# System information
fractal-gen system-info
fractal-gen benchmark
```

### Configuration Management

```bash
# Create configuration template
fractal-gen init-config --output config.yaml --with-examples

# Validate configuration
fractal-gen validate-config config.yaml

# List available options
fractal-gen list-fractals
fractal-gen list-palettes  
fractal-gen list-presets
```

### Render Options

- `--width, -w`: Image width (default: 1920)
- `--height, -h`: Image height (default: 1080)  
- `--bounds`: Complex plane bounds "xmin,xmax,ymin,ymax"
- `--max-iter`: Maximum iterations (default: 1000)
- `--precision`: Numerical precision (single/double/quad)
- `--palette`: Color palette name
- `--algorithm`: Coloring algorithm
- `--no-gpu`: Disable GPU acceleration
- `--processes`: Number of processes for parallel rendering

## Dependencies

### Core Requirements
- Python 3.8+
- NumPy >= 1.20.0
- Pillow >= 8.0.0  
- PyYAML >= 5.4.0
- Click >= 8.0.0

### Optional Dependencies
- **GPU**: CuPy >= 10.0.0 (for CUDA support)
- **Performance**: Numba >= 0.56.0 (for JIT compilation)
- **High Precision**: mpmath >= 1.2.0 (for arbitrary precision)
- **Video**: imageio[ffmpeg] >= 2.10.0 (for animation export)
- **Scientific**: SciPy >= 1.7.0, matplotlib >= 3.3.0
- **Web Interface**: Streamlit >= 1.0.0, Flask >= 2.0.0

## Examples

### Gallery Generation

```python
# Generate a gallery of different fractals
from fractal_generator.examples.gallery import create_gallery

create_gallery(
    output_dir='gallery/',
    fractals=['mandelbrot', 'julia', 'burning_ship'],
    sizes=[(800, 600), (1920, 1080)],
    palettes=['hot', 'viridis', 'plasma']
)
```

### Deep Zoom Sequence

```python
# Create deep zoom into Mandelbrot set
config = RenderConfig(precision='quad', max_iterations=5000)
renderer = FractalRenderer(config)

zoom_center = complex(-0.7269, 0.1889)
for i, zoom_level in enumerate(np.logspace(1, 6, 50)):
    bounds = calculate_zoom_bounds(zoom_center, zoom_level)
    config.bounds = bounds
    
    image = renderer.render(MandelbrotSet(), f'deep_zoom_{i:03d}.png')
```

### Scientific Analysis

```python
# Generate data for fractal dimension analysis
from fractal_generator.analysis import FractalAnalyzer

analyzer = FractalAnalyzer()
fractal = MandelbrotSet()

# Calculate boundary complexity
complexity = analyzer.boundary_complexity(fractal, resolution=2048)

# Generate escape time statistics
stats = analyzer.escape_statistics(fractal, num_samples=1000000)
```

## API Reference

### Main Classes

- **`FractalRenderer`**: Main rendering engine
- **`RenderConfig`**: Configuration management
- **`FractalExplorer`**: Interactive exploration
- **`BatchRenderer`**: Batch processing
- **`AnimationSequencer`**: Animation generation

### Fractal Types

- **`MandelbrotSet`**: Classic Mandelbrot fractal
- **`JuliaSet`**: Julia set fractals
- **`BurningShip`**: Burning Ship fractal
- **`Multibrot`**: Generalized Mandelbrot with custom powers
- **`CustomFractal`**: User-defined iteration functions

### Coloring and Rendering

- **`ColoringEngine`**: Color management and algorithms
- **`Palette`**: Color palette handling
- **`ImageExporter`**: High-quality image export
- **`ImageProcessor`**: Post-processing operations

## Performance Guide

### Optimal Settings by Use Case

#### Preview/Development
```python
config = RenderConfig(
    width=800, height=600,
    max_iterations=100,
    use_gpu=False,
    precision='single'
)
```

#### High Quality Prints
```python  
config = RenderConfig(
    width=7680, height=4320,  # 8K resolution
    max_iterations=2000,
    antialiasing=2,
    precision='double',
    output_format='png'
)
```

#### Deep Zoom Research
```python
config = RenderConfig(
    precision=100,  # 100 decimal places
    max_iterations=10000,
    use_multiprocessing=True,
    tile_size=128
)
```

### Memory Management

- **Large Images**: Use tile-based rendering to manage memory
- **Deep Zooms**: Enable arbitrary precision only when needed
- **Batch Jobs**: Process sequentially to avoid memory accumulation
- **GPU Memory**: Monitor VRAM usage for very large renders

### Troubleshooting

#### Common Issues

1. **Out of Memory**: Reduce tile size or image resolution
2. **Slow Performance**: Enable GPU/Numba acceleration  
3. **Numerical Artifacts**: Increase precision for deep zooms
4. **Import Errors**: Install optional dependencies as needed

#### Debug Mode

```bash
fractal-gen render mandelbrot test.png --verbose
```

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup

```bash
git clone https://github.com/username/fractal-generator.git
cd fractal-generator
pip install -e .[dev]
```

### Testing

```bash
pytest tests/
pytest --cov=fractal_generator tests/  # With coverage
```

### Code Style

```bash  
black fractal_generator/
flake8 fractal_generator/
mypy fractal_generator/
```

### Adding New Fractals

1. Inherit from `FractalType`
2. Implement `compute()` method
3. Add to `FractalRegistry`
4. Add tests and documentation

```python
class MyFractal(FractalType):
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        # Implement your fractal algorithm
        pass
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        return (-2, 2, -2, 2)

# Register the fractal
FractalRegistry.register('my_fractal', MyFractal)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Mathematical Foundation**: Based on the work of Benoit Mandelbrot and Gaston Julia
- **Performance Libraries**: NumPy, Numba, and CuPy teams for excellent numerical computing tools
- **Visualization**: Matplotlib and Pillow for graphics and color management
- **Community**: Thanks to all contributors and the fractal mathematics community

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{fractal_generator,
  title={Fractal Generator: High-Performance Fractal Image Generation},
  author={Your Name},
  year={2024},
  url={https://github.com/username/fractal-generator}
}
```

## Changelog

### Version 1.0.0
- Initial release
- Support for Mandelbrot, Julia, Burning Ship, and Multibrot fractals
- GPU acceleration via CuPy
- Numba JIT compilation
- Multiprocessing tile-based rendering
- Arbitrary precision support
- Advanced coloring algorithms
- Configuration management system
- Command-line interface
- Interactive exploration tools
- Animation and video export
- Comprehensive documentation

## Roadmap

### Upcoming Features

- **Web Interface**: Browser-based fractal explorer
- **3D Fractals**: Support for 3D fractal types (Mandelbulb, etc.)
- **Real-time Rendering**: Interactive real-time parameter adjustment
- **Cloud Integration**: Distributed rendering across multiple machines
- **Machine Learning**: AI-assisted parameter optimization
- **Virtual Reality**: VR fractal exploration
- **Additional Fractals**: Newton fractals, IFS fractals, L-systems

### Performance Improvements

- **Optimized Kernels**: Hand-tuned CUDA kernels for maximum GPU performance
- **Distributed Computing**: Multi-GPU and cluster support
- **Advanced Algorithms**: Perturbation theory for ultra-deep zooms
- **Memory Optimization**: Streaming algorithms for massive images

## Support

### Getting Help

- **Email**: m00800196@gmail.com

### Commercial Support

Professional support and custom development services available. Contact us for:

- Custom fractal algorithms
- Performance optimization consulting
- Integration assistance
- Training and workshops

## Examples and Tutorials

### Tutorial: Creating Your First Fractal

```python
# 1. Import required modules
from fractal_generator import FractalRenderer, MandelbrotSet, RenderConfig

# 2. Configure the renderer
config = RenderConfig(
    width=1920,           # Image width
    height=1080,          # Image height  
    max_iterations=1000,  # Detail level
    color_palette='hot',  # Color scheme
    bounds=(-2, 1, -1.5, 1.5)  # View area
)

# 3. Create renderer and fractal
renderer = FractalRenderer(config)
fractal = MandelbrotSet()

# 4. Render the fractal
image = renderer.render(fractal, 'my_first_fractal.png')
print("Fractal saved as my_first_fractal.png")
```

### Tutorial: Exploring Julia Sets

```python
from fractal_generator import JuliaSet, JuliaParameters

# Try different Julia constants
interesting_points = [
    (-0.75, 0.1),    # Dragon
    (-0.4, 0.6),     # Spiral  
    (-0.8, 0.156),   # Lightning
    (-0.123, 0.745), # Rabbit
]

for i, (real, imag) in enumerate(interesting_points):
    # Create Julia set with specific constant
    params = JuliaParameters(c_real=real, c_imag=imag)
    julia = JuliaSet(params)
    
    # Render
    image = renderer.render(julia, f'julia_{i}.png')
    print(f"Created julia_{i}.png with c={real}+{imag}i")
```

### Tutorial: Animation Creation

```python
from fractal_generator.tools.animation import AnimationSequencer
import numpy as np

# Create zoom animation
sequencer = AnimationSequencer(config)

# Zoom into an interesting point
zoom_center = complex(-0.5, 0.0)
zoom_factors = np.logspace(0, 2, 60)  # 60 frames, 1x to 100x zoom

sequencer.create_zoom_sequence(
    fractal=MandelbrotSet(),
    center=zoom_center,
    zoom_factor=1.5,  # Per frame
    frames=60,
    output_dir='zoom_animation/'
)

# Convert to video
sequencer.create_video_from_images(
    'zoom_animation/', 
    'mandelbrot_zoom.mp4', 
    fps=30
)
```

### Tutorial: Custom Color Palettes

```python
from fractal_generator.rendering.coloring import Palette, ColorRGB

# Define custom colors
sunset_colors = [
    ColorRGB(0.1, 0.1, 0.2),    # Dark blue
    ColorRGB(0.3, 0.2, 0.4),    # Purple
    ColorRGB(0.8, 0.4, 0.2),    # Orange
    ColorRGB(1.0, 0.8, 0.3),    # Yellow
    ColorRGB(1.0, 1.0, 0.9),    # Light yellow
]

# Create palette
sunset_palette = Palette(sunset_colors, name="Sunset")

# Add to renderer
renderer.coloring_engine.add_palette("sunset", sunset_palette)

# Use in rendering
config.color_palette = "sunset"
image = renderer.render(fractal, 'sunset_fractal.png')
```

### Tutorial: High-Resolution Printing

```python
# Configuration for high-quality prints
print_config = RenderConfig(
    width=7680,           # 8K width
    height=4320,          # 8K height
    max_iterations=3000,  # High detail
    antialiasing=2,       # Smooth edges
    precision='double',   # Numerical accuracy
    output_format='tiff', # Lossless format
    use_multiprocessing=True,  # Parallel processing
    tile_size=512,        # Memory management
)

# Create high-resolution renderer
print_renderer = FractalRenderer(print_config)

# Render for printing
print_image = print_renderer.render(
    MandelbrotSet(), 
    'mandelbrot_8k_print.tiff'
)

print("High-resolution image ready for printing!")
```

## FAQ

**Q: Which fractal type should I start with?**
A: The Mandelbrot set is the most recognizable and offers great exploration opportunities. Julia sets provide beautiful variations with different constants.

**Q: How do I choose the right number of iterations?**
A: Start with 1000 iterations for general use. Increase for more detail in complex areas, decrease for faster previews. Deep zooms may need 5000+ iterations.

**Q: When should I use GPU acceleration?**
A: GPU acceleration is beneficial for images larger than 500,000 pixels. For smaller images, CPU rendering is often faster due to GPU setup overhead.

**Q: How much memory do I need for high-resolution images?**
A: Rough estimate: Width × Height × 32 bytes per pixel. A 4K image needs ~250MB, 8K needs ~1GB. Tile-based rendering reduces peak memory usage.

**Q: Can I create commercial art with this library?**
A: Yes, the MIT license allows commercial use. The fractals themselves are mathematical objects and cannot be copyrighted.

**Q: How do I report bugs or request features?**
A: Use GitHub Issues for bugs and feature requests. Include system information and reproduction steps for bugs.

**Q: Is this library suitable for scientific research?**
A: Yes, the arbitrary precision support and comprehensive parameter control make it suitable for mathematical research into fractal properties.

---

## Quick Reference

### Essential Commands
```bash
# Basic render
fractal-gen render mandelbrot output.png

# High quality  
fractal-gen render mandelbrot output.png --preset high_quality

# Interactive
fractal-gen explore mandelbrot

# Animation
fractal-gen animate mandelbrot anim/ --frames 60
```

### Key Python Classes
```python
from fractal_generator import (
    FractalRenderer,      # Main rendering engine
    RenderConfig,         # Configuration
    MandelbrotSet,        # Mandelbrot fractal  
    JuliaSet,            # Julia set fractal
    FractalExplorer,     # Interactive exploration
    BatchRenderer,       # Batch processing
)
```

### Performance Tips
- Use GPU for images > 500K pixels
- Enable multiprocessing for images > 1M pixels  
- Use arbitrary precision only for extreme zooms
- Reduce tile size if running out of memory
- Start with lower iterations for parameter exploration

### Common Bounds
- **Mandelbrot full view**: `(-2.5, 1.0, -1.25, 1.25)`
- **Mandelbrot detail**: `(-0.8, -0.4, -0.2, 0.2)` 
- **Julia sets**: `(-2.0, 2.0, -2.0, 2.0)`
- **Burning Ship**: `(-2.5, 1.5, -2.0, 1.0)`

Ready to create beautiful fractals! 🎨✨
