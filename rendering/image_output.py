"""
Image export and format handling for fractal rendering.

This module provides comprehensive image export capabilities supporting
various formats (PNG, TIFF, JPEG) with metadata embedding and quality
control options.
"""

import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import logging
from datetime import datetime

try:
    from PIL import Image, PngImagePlugin, TiffImagePlugin
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("Pillow not available - image export disabled")

logger = logging.getLogger(__name__)


@dataclass
class RenderMetadata:
    """Metadata for fractal renders."""
    
    # Fractal parameters
    fractal_type: str
    bounds: Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    resolution: Tuple[int, int]  # width, height
    max_iterations: int
    escape_radius: float
    
    # Rendering parameters
    coloring_algorithm: str
    color_palette: str
    precision: str
    
    # Timing and performance
    render_time_seconds: float
    tiles_used: bool = False
    gpu_accelerated: bool = False
    
    # Generation info
    timestamp: str = ""
    software_version: str = "1.0.0"
    
    # Fractal-specific parameters
    fractal_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        """Set default timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        
        if self.fractal_parameters is None:
            self.fractal_parameters = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RenderMetadata':
        """Create metadata from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ImageExporter:
    """High-quality image export with metadata support."""
    
    def __init__(self):
        """Initialize image exporter."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow required for image export")
        
        self.supported_formats = {
            '.png': self._save_png,
            '.tiff': self._save_tiff,
            '.tif': self._save_tiff,
            '.jpg': self._save_jpeg,
            '.jpeg': self._save_jpeg,
        }
    
    def save_image(self, image_array: np.ndarray, filepath: Path,
                   metadata: Optional[RenderMetadata] = None,
                   quality: int = 95, compression: Optional[str] = None) -> None:
        """
        Save RGB image array to file with metadata.
        
        Args:
            image_array: RGB image array (height, width, 3) with values 0-1
            filepath: Output file path
            metadata: Render metadata to embed
            quality: JPEG quality (1-100)
            compression: Compression method for TIFF
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()
        
        if suffix not in self.supported_formats:
            supported = ', '.join(self.supported_formats.keys())
            raise ValueError(f"Unsupported format '{suffix}'. Supported: {supported}")
        
        # Validate and convert image array
        image_array = self._prepare_image_array(image_array)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Save with format-specific method
        save_method = self.supported_formats[suffix]
        save_method(pil_image, filepath, metadata, quality, compression)
        
        logger.info(f"Saved image: {filepath} ({pil_image.size[0]}x{pil_image.size[1]})")
    
    def _prepare_image_array(self, image_array: np.ndarray) -> np.ndarray:
        """Prepare and validate image array for export."""
        # Ensure we have the right shape
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError(f"Expected RGB image array (H, W, 3), got {image_array.shape}")
        
        # Convert to 8-bit
        if image_array.dtype != np.uint8:
            # Assume values are in 0-1 range if float
            if np.issubdtype(image_array.dtype, np.floating):
                # Clip to valid range and convert
                image_array = np.clip(image_array, 0.0, 1.0)
                image_array = (image_array * 255).astype(np.uint8)
            else:
                # Assume already in 0-255 range
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        return image_array
    
    def _save_png(self, pil_image: Image.Image, filepath: Path,
                  metadata: Optional[RenderMetadata], quality: int, compression: Optional[str]) -> None:
        """Save as PNG with metadata."""
        pnginfo = PngImagePlugin.PngInfo()
        
        if metadata:
            # Add standard PNG text chunks
            pnginfo.add_text("Title", f"Fractal: {metadata.fractal_type}")
            pnginfo.add_text("Description", f"Generated fractal image")
            pnginfo.add_text("Software", f"FractalGenerator v{metadata.software_version}")
            pnginfo.add_text("Creation Time", metadata.timestamp)
            
            # Add comprehensive metadata as JSON
            pnginfo.add_text("FractalMetadata", metadata.to_json())
        
        # PNG compression levels: 0 (no compression) to 9 (max compression)
        compress_level = 6  # Default
        if compression:
            if compression.lower() in ['none', '0']:
                compress_level = 0
            elif compression.lower() in ['fast', 'low']:
                compress_level = 1
            elif compression.lower() in ['high', 'max']:
                compress_level = 9
        
        pil_image.save(filepath, "PNG", pnginfo=pnginfo, compress_level=compress_level)
    
    def _save_tiff(self, pil_image: Image.Image, filepath: Path,
                   metadata: Optional[RenderMetadata], quality: int, compression: Optional[str]) -> None:
        """Save as TIFF with metadata."""
        # Prepare TIFF tags
        tiff_tags = {}
        
        if metadata:
            # Standard TIFF tags
            tiff_tags[270] = f"Fractal: {metadata.fractal_type}"  # ImageDescription
            tiff_tags[305] = f"FractalGenerator v{metadata.software_version}"  # Software
            tiff_tags[306] = metadata.timestamp  # DateTime
            
            # Custom metadata in TIFF tag
            tiff_tags[42000] = metadata.to_json()  # Custom tag for fractal metadata
        
        # Compression options
        if compression is None:
            compression = 'lzw'  # Default for TIFF
        
        compression_map = {
            'none': None,
            'lzw': 'lzw',
            'jpeg': 'jpeg',
            'deflate': 'tiff_deflate',
            'zip': 'tiff_deflate',
        }
        
        tiff_compression = compression_map.get(compression.lower(), 'lzw')
        
        save_kwargs = {'format': 'TIFF'}
        if tiff_compression:
            save_kwargs['compression'] = tiff_compression
        
        # Add quality for JPEG compression in TIFF
        if tiff_compression == 'jpeg':
            save_kwargs['quality'] = quality
        
        pil_image.save(filepath, **save_kwargs)
        
        # Add custom tags after saving (PIL limitation workaround)
        if tiff_tags:
            self._add_tiff_tags(filepath, tiff_tags)
    
    def _save_jpeg(self, pil_image: Image.Image, filepath: Path,
                   metadata: Optional[RenderMetadata], quality: int, compression: Optional[str]) -> None:
        """Save as JPEG with metadata."""
        # JPEG doesn't support as rich metadata as PNG/TIFF
        # We'll save a companion JSON file for full metadata
        
        pil_image.save(filepath, "JPEG", quality=quality, optimize=True)
        
        if metadata:
            # Save metadata as companion JSON file
            json_path = filepath.with_suffix('.json')
            with open(json_path, 'w') as f:
                f.write(metadata.to_json())
            logger.info(f"Saved metadata: {json_path}")
    
    def _add_tiff_tags(self, filepath: Path, tags: Dict[int, str]) -> None:
        """Add custom TIFF tags (simplified implementation)."""
        # Note: This is a basic implementation
        # Full TIFF tag manipulation requires more sophisticated tools
        try:
            # Try to add tags using PIL's TIFF handling
            with Image.open(filepath) as img:
                for tag, value in tags.items():
                    try:
                        img.tag_v2[tag] = value
                    except Exception as e:
                        logger.debug(f"Could not add TIFF tag {tag}: {e}")
        except Exception as e:
            logger.debug(f"Could not modify TIFF tags: {e}")
    
    def save_raw_data(self, image_array: np.ndarray, filepath: Path,
                     metadata: Optional[RenderMetadata] = None) -> None:
        """
        Save raw image data as NumPy array.
        
        Args:
            image_array: Image array to save
            filepath: Output file path (.npy)
            metadata: Metadata to save alongside
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.npy':
            filepath = filepath.with_suffix('.npy')
        
        # Save array
        np.save(filepath, image_array)
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                f.write(metadata.to_json())
        
        logger.info(f"Saved raw data: {filepath}")
    
    def load_raw_data(self, filepath: Path) -> Tuple[np.ndarray, Optional[RenderMetadata]]:
        """
        Load raw image data and metadata.
        
        Args:
            filepath: Input file path (.npy)
            
        Returns:
            Tuple of (image_array, metadata)
        """
        filepath = Path(filepath)
        
        # Load array
        image_array = np.load(filepath)
        
        # Try to load metadata
        metadata = None
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = RenderMetadata.from_json(f.read())
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_path}: {e}")
        
        return image_array, metadata
    
    def create_image_sequence(self, image_arrays: list, output_dir: Path,
                             base_name: str = "frame", format: str = "png",
                             metadata_list: Optional[list] = None,
                             padding: int = 6) -> list:
        """
        Save a sequence of images with sequential numbering.
        
        Args:
            image_arrays: List of image arrays to save
            output_dir: Output directory
            base_name: Base filename
            format: Image format
            metadata_list: Optional list of metadata for each frame
            padding: Number of digits for frame numbering
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, image_array in enumerate(image_arrays):
            # Generate filename with padding
            frame_num = str(i).zfill(padding)
            filename = f"{base_name}_{frame_num}.{format}"
            filepath = output_dir / filename
            
            # Get metadata for this frame
            metadata = None
            if metadata_list and i < len(metadata_list):
                metadata = metadata_list[i]
            
            # Save image
            self.save_image(image_array, filepath, metadata)
            saved_paths.append(filepath)
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
    
    def extract_metadata_from_image(self, filepath: Path) -> Optional[RenderMetadata]:
        """
        Extract fractal metadata from saved image.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Extracted metadata or None
        """
        filepath = Path(filepath)
        
        try:
            with Image.open(filepath) as img:
                # Try PNG metadata first
                if hasattr(img, 'text') and 'FractalMetadata' in img.text:
                    return RenderMetadata.from_json(img.text['FractalMetadata'])
                
                # Try TIFF metadata
                if hasattr(img, 'tag_v2') and 42000 in img.tag_v2:
                    return RenderMetadata.from_json(img.tag_v2[42000])
                
                # For JPEG, look for companion JSON file
                if filepath.suffix.lower() in ['.jpg', '.jpeg']:
                    json_path = filepath.with_suffix('.json')
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            return RenderMetadata.from_json(f.read())
        
        except Exception as e:
            logger.warning(f"Could not extract metadata from {filepath}: {e}")
        
        return None
    
    def get_image_info(self, filepath: Path) -> Dict[str, Any]:
        """
        Get comprehensive information about an image file.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Dictionary with image information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")
        
        info = {
            'filepath': str(filepath),
            'size_bytes': filepath.stat().st_size,
            'format': None,
            'dimensions': None,
            'mode': None,
            'has_fractal_metadata': False,
            'fractal_metadata': None,
        }
        
        try:
            with Image.open(filepath) as img:
                info['format'] = img.format
                info['dimensions'] = img.size
                info['mode'] = img.mode
                
                # Check for fractal metadata
                metadata = self.extract_metadata_from_image(filepath)
                if metadata:
                    info['has_fractal_metadata'] = True
                    info['fractal_metadata'] = metadata.to_dict()
        
        except Exception as e:
            logger.error(f"Error reading image {filepath}: {e}")
            info['error'] = str(e)
        
        return info


class ImageProcessor:
    """Post-processing operations for fractal images."""
    
    def __init__(self):
        """Initialize image processor."""
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow required for image processing")
    
    def resize_image(self, image_array: np.ndarray, new_size: Tuple[int, int],
                    method: str = 'lanczos') -> np.ndarray:
        """
        Resize image using high-quality resampling.
        
        Args:
            image_array: Input image array
            new_size: Target size (width, height)
            method: Resampling method ('lanczos', 'bicubic', 'bilinear', 'nearest')
            
        Returns:
            Resized image array
        """
        # Convert to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Resampling method mapping
        method_map = {
            'lanczos': Image.Resampling.LANCZOS,
            'bicubic': Image.Resampling.BICUBIC,
            'bilinear': Image.Resampling.BILINEAR,
            'nearest': Image.Resampling.NEAREST,
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown resampling method: {method}")
        
        # Resize
        resized_pil = pil_image.resize(new_size, method_map[method])
        
        # Convert back to numpy array
        return np.array(resized_pil)
    
    def apply_gamma_correction(self, image_array: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to image.
        
        Args:
            image_array: Input image array (0-1 range)
            gamma: Gamma value (>1 darkens, <1 brightens)
            
        Returns:
            Gamma-corrected image array
        """
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        
        # Ensure input is in 0-1 range
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float64) / 255.0
        
        # Apply gamma correction
        corrected = np.power(image_array, gamma)
        
        return np.clip(corrected, 0, 1)
    
    def adjust_contrast_brightness(self, image_array: np.ndarray, 
                                 contrast: float = 1.0, brightness: float = 0.0) -> np.ndarray:
        """
        Adjust image contrast and brightness.
        
        Args:
            image_array: Input image array (0-1 range)
            contrast: Contrast multiplier (1.0 = no change)
            brightness: Brightness offset (-1 to 1)
            
        Returns:
            Adjusted image array
        """
        # Ensure input is in 0-1 range
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float64) / 255.0
        
        # Apply contrast and brightness
        adjusted = contrast * image_array + brightness
        
        return np.clip(adjusted, 0, 1)
    
    def apply_unsharp_mask(self, image_array: np.ndarray, radius: float = 1.0,
                          amount: float = 1.0, threshold: float = 0.0) -> np.ndarray:
        """
        Apply unsharp mask for image sharpening.
        
        Args:
            image_array: Input image array
            radius: Blur radius for mask
            amount: Strength of sharpening
            threshold: Minimum difference threshold
            
        Returns:
            Sharpened image array
        """
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            logger.warning("scipy not available - skipping unsharp mask")
            return image_array
        
        # Ensure input is in 0-1 range
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float64) / 255.0
        
        # Create blurred version
        blurred = np.zeros_like(image_array)
        for channel in range(image_array.shape[2]):
            blurred[:, :, channel] = gaussian_filter(image_array[:, :, channel], sigma=radius)
        
        # Calculate mask
        mask = image_array - blurred
        
        # Apply threshold
        if threshold > 0:
            mask = np.where(np.abs(mask) >= threshold, mask, 0)
        
        # Apply sharpening
        sharpened = image_array + amount * mask
        
        return np.clip(sharpened, 0, 1)
    
    def create_thumbnail(self, image_array: np.ndarray, max_size: int = 256) -> np.ndarray:
        """
        Create thumbnail maintaining aspect ratio.
        
        Args:
            image_array: Input image array
            max_size: Maximum dimension size
            
        Returns:
            Thumbnail image array
        """
        height, width = image_array.shape[:2]
        
        # Calculate new dimensions
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        return self.resize_image(image_array, (new_width, new_height))
    
    def add_watermark(self, image_array: np.ndarray, text: str,
                     position: str = 'bottom-right', opacity: float = 0.7,
                     font_size: int = 24) -> np.ndarray:
        """
        Add text watermark to image.
        
        Args:
            image_array: Input image array
            text: Watermark text
            position: Position ('bottom-right', 'bottom-left', etc.)
            opacity: Watermark opacity (0-1)
            font_size: Font size in points
            
        Returns:
            Watermarked image array
        """
        # Convert to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image_array, mode='RGB')
        
        # Create watermark
        try:
            from PIL import ImageDraw, ImageFont
            
            # Create overlay
            overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to use a decent font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Calculate text position
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # Rough estimate without font
                text_width = len(text) * font_size * 0.6
                text_height = font_size
            
            margin = 10
            
            if position == 'bottom-right':
                x = pil_image.width - text_width - margin
                y = pil_image.height - text_height - margin
            elif position == 'bottom-left':
                x = margin
                y = pil_image.height - text_height - margin
            elif position == 'top-right':
                x = pil_image.width - text_width - margin
                y = margin
            elif position == 'top-left':
                x = margin
                y = margin
            else:  # center
                x = (pil_image.width - text_width) // 2
                y = (pil_image.height - text_height) // 2
            
            # Draw text with opacity
            alpha = int(opacity * 255)
            if font:
                draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))
            else:
                draw.text((x, y), text, fill=(255, 255, 255, alpha))
            
            # Composite with original image
            watermarked = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
            watermarked = watermarked.convert('RGB')
            
            return np.array(watermarked)
        
        except Exception as e:
            logger.warning(f"Could not add watermark: {e}")
            return image_array
