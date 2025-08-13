"""
Fractal type definitions and parameter management.

This module defines various fractal types as configurable classes,
providing a plugin-style architecture for different fractal algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging

from .math_functions import FractalIterator, IterationResult, ComplexPlane

logger = logging.getLogger(__name__)


@dataclass
class FractalParameters:
    """Base class for fractal parameters with validation."""
    
    def validate(self) -> None:
        """Validate parameter values."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FractalParameters':
        """Create parameters from dictionary."""
        return cls(**data)


class FractalType(ABC):
    """Abstract base class for fractal types."""
    
    def __init__(self, name: str, parameters: FractalParameters):
        """
        Initialize fractal type.
        
        Args:
            name: Human-readable name for the fractal
            parameters: Fractal-specific parameters
        """
        self.name = name
        self.parameters = parameters
        self.parameters.validate()
    
    @abstractmethod
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """
        Compute fractal iterations for the given complex plane.
        
        Args:
            plane: Complex plane definition
            iterator: Fractal iterator instance
            
        Returns:
            IterationResult containing iteration data
        """
        pass
    
    @abstractmethod
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds (xmin, xmax, ymin, ymax)."""
        pass
    
    def get_description(self) -> str:
        """Get a description of this fractal type."""
        return f"{self.name} fractal"


@dataclass
class MandelbrotParameters(FractalParameters):
    """Parameters for Mandelbrot set generation."""
    
    z0_real: float = 0.0
    z0_imag: float = 0.0
    
    def validate(self) -> None:
        """Validate Mandelbrot parameters."""
        if not isinstance(self.z0_real, (int, float)):
            raise ValueError("z0_real must be numeric")
        if not isinstance(self.z0_imag, (int, float)):
            raise ValueError("z0_imag must be numeric")


class MandelbrotSet(FractalType):
    """Mandelbrot set fractal implementation."""
    
    def __init__(self, parameters: Optional[MandelbrotParameters] = None):
        """
        Initialize Mandelbrot set.
        
        Args:
            parameters: Mandelbrot-specific parameters
        """
        if parameters is None:
            parameters = MandelbrotParameters()
        super().__init__("Mandelbrot", parameters)
    
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """Compute Mandelbrot set iterations."""
        c = plane.create_complex_array(iterator.dtype)
        
        # Initial z value (usually 0, but configurable)
        z0 = None
        if self.parameters.z0_real != 0.0 or self.parameters.z0_imag != 0.0:
            z0 = np.full_like(c, complex(self.parameters.z0_real, self.parameters.z0_imag))
        
        return iterator.mandelbrot_iteration(c, z0)
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds for Mandelbrot set."""
        return (-2.5, 1.0, -1.25, 1.25)
    
    def get_description(self) -> str:
        """Get description of Mandelbrot set."""
        return ("Mandelbrot set: z_{n+1} = z_n^2 + c, where c is the complex coordinate "
                "and z_0 = " + str(complex(self.parameters.z0_real, self.parameters.z0_imag)))


@dataclass
class JuliaParameters(FractalParameters):
    """Parameters for Julia set generation."""
    
    c_real: float = -0.75
    c_imag: float = 0.1
    
    def validate(self) -> None:
        """Validate Julia parameters."""
        if not isinstance(self.c_real, (int, float)):
            raise ValueError("c_real must be numeric")
        if not isinstance(self.c_imag, (int, float)):
            raise ValueError("c_imag must be numeric")
    
    @property
    def c(self) -> complex:
        """Get the Julia constant as a complex number."""
        return complex(self.c_real, self.c_imag)


class JuliaSet(FractalType):
    """Julia set fractal implementation."""
    
    def __init__(self, parameters: Optional[JuliaParameters] = None):
        """
        Initialize Julia set.
        
        Args:
            parameters: Julia-specific parameters
        """
        if parameters is None:
            parameters = JuliaParameters()
        super().__init__("Julia", parameters)
    
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """Compute Julia set iterations."""
        z = plane.create_complex_array(iterator.dtype)
        return iterator.julia_iteration(z, self.parameters.c)
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds for Julia set."""
        return (-2.0, 2.0, -2.0, 2.0)
    
    def get_description(self) -> str:
        """Get description of Julia set."""
        return f"Julia set: z_{{n+1}} = z_n^2 + c, where c = {self.parameters.c} and z_0 is the complex coordinate"


@dataclass
class BurningShipParameters(FractalParameters):
    """Parameters for Burning Ship fractal."""
    
    def validate(self) -> None:
        """Validate Burning Ship parameters."""
        pass  # No specific parameters to validate


class BurningShip(FractalType):
    """Burning Ship fractal implementation."""
    
    def __init__(self, parameters: Optional[BurningShipParameters] = None):
        """
        Initialize Burning Ship fractal.
        
        Args:
            parameters: Burning Ship parameters
        """
        if parameters is None:
            parameters = BurningShipParameters()
        super().__init__("Burning Ship", parameters)
    
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """Compute Burning Ship fractal iterations."""
        c = plane.create_complex_array(iterator.dtype)
        return iterator.burning_ship_iteration(c)
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds for Burning Ship."""
        return (-2.5, 1.5, -2.0, 1.0)
    
    def get_description(self) -> str:
        """Get description of Burning Ship fractal."""
        return "Burning Ship: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c"


@dataclass
class MultibrotParameters(FractalParameters):
    """Parameters for Multibrot fractal."""
    
    power: float = 3.0
    
    def validate(self) -> None:
        """Validate Multibrot parameters."""
        if not isinstance(self.power, (int, float)):
            raise ValueError("power must be numeric")
        if self.power == 0:
            raise ValueError("power cannot be zero")


class Multibrot(FractalType):
    """Multibrot fractal implementation."""
    
    def __init__(self, parameters: Optional[MultibrotParameters] = None):
        """
        Initialize Multibrot fractal.
        
        Args:
            parameters: Multibrot parameters
        """
        if parameters is None:
            parameters = MultibrotParameters()
        super().__init__("Multibrot", parameters)
    
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """Compute Multibrot fractal iterations."""
        c = plane.create_complex_array(iterator.dtype)
        return iterator.multibrot_iteration(c, self.parameters.power)
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds for Multibrot."""
        # Bounds depend on the power
        if self.parameters.power < 2:
            scale = 3.0
        elif self.parameters.power > 4:
            scale = 1.5
        else:
            scale = 2.0
        return (-scale, scale, -scale, scale)
    
    def get_description(self) -> str:
        """Get description of Multibrot fractal."""
        return f"Multibrot: z_{{n+1}} = z_n^{self.parameters.power} + c"


@dataclass
class CustomFractalParameters(FractalParameters):
    """Parameters for custom fractal functions."""
    
    function_name: str = "custom"
    description: str = "Custom fractal"
    recommended_bounds: Tuple[float, float, float, float] = (-2.0, 2.0, -2.0, 2.0)
    
    def validate(self) -> None:
        """Validate custom fractal parameters."""
        if not isinstance(self.function_name, str):
            raise ValueError("function_name must be a string")
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")


class CustomFractal(FractalType):
    """Custom fractal implementation with user-defined iteration function."""
    
    def __init__(self, iteration_func: Callable[[np.ndarray], np.ndarray],
                 parameters: Optional[CustomFractalParameters] = None):
        """
        Initialize custom fractal.
        
        Args:
            iteration_func: Function that takes complex array and returns next iteration
            parameters: Custom fractal parameters
        """
        if parameters is None:
            parameters = CustomFractalParameters()
        super().__init__("Custom", parameters)
        self.iteration_func = iteration_func
    
    def compute(self, plane: ComplexPlane, iterator: FractalIterator) -> IterationResult:
        """Compute custom fractal iterations."""
        initial_values = plane.create_complex_array(iterator.dtype)
        return iterator.custom_iteration(initial_values, self.iteration_func)
    
    def get_recommended_bounds(self) -> Tuple[float, float, float, float]:
        """Get recommended viewing bounds for custom fractal."""
        return self.parameters.recommended_bounds
    
    def get_description(self) -> str:
        """Get description of custom fractal."""
        return self.parameters.description


class FractalRegistry:
    """Registry for managing available fractal types."""
    
    _fractals: Dict[str, type] = {
        'mandelbrot': MandelbrotSet,
        'julia': JuliaSet,
        'burning_ship': BurningShip,
        'multibrot': Multibrot,
        'custom': CustomFractal,
    }
    
    @classmethod
    def register(cls, name: str, fractal_class: type) -> None:
        """
        Register a new fractal type.
        
        Args:
            name: Unique identifier for the fractal
            fractal_class: Class implementing the fractal
        """
        if not issubclass(fractal_class, FractalType):
            raise ValueError(f"Fractal class must inherit from FractalType")
        cls._fractals[name.lower()] = fractal_class
        logger.info(f"Registered fractal type: {name}")
    
    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a fractal class by name.
        
        Args:
            name: Fractal identifier
            
        Returns:
            Fractal class
        """
        fractal_class = cls._fractals.get(name.lower())
        if fractal_class is None:
            available = ', '.join(cls._fractals.keys())
            raise ValueError(f"Unknown fractal type '{name}'. Available: {available}")
        return fractal_class
    
    @classmethod
    def list_fractals(cls) -> Dict[str, str]:
        """Get a dictionary of available fractals and their descriptions."""
        result = {}
        for name, fractal_class in cls._fractals.items():
            if fractal_class != CustomFractal:
                # Create a temporary instance to get description
                if name == 'mandelbrot':
                    temp = fractal_class()
                elif name == 'julia':
                    temp = fractal_class()
                elif name == 'burning_ship':
                    temp = fractal_class()
                elif name == 'multibrot':
                    temp = fractal_class()
                else:
                    continue
                result[name] = temp.get_description()
        return result
    
    @classmethod
    def create_fractal(cls, name: str, **kwargs) -> FractalType:
        """
        Create a fractal instance with the given parameters.
        
        Args:
            name: Fractal type name
            **kwargs: Parameters for the fractal
            
        Returns:
            Configured fractal instance
        """
        fractal_class = cls.get(name)
        
        if name.lower() == 'custom':
            if 'iteration_func' not in kwargs:
                raise ValueError("Custom fractal requires 'iteration_func' parameter")
            return fractal_class(**kwargs)
        else:
            # Create parameters object for built-in fractals
            param_classes = {
                'mandelbrot': MandelbrotParameters,
                'julia': JuliaParameters,
                'burning_ship': BurningShipParameters,
                'multibrot': MultibrotParameters,
            }
            
            param_class = param_classes.get(name.lower())
            if param_class and kwargs:
                parameters = param_class(**kwargs)
                return fractal_class(parameters)
            else:
                return fractal_class()


# Predefined interesting Julia set constants
JULIA_PRESETS = {
    'dragon': JuliaParameters(c_real=-0.75, c_imag=0.1),
    'spiral': JuliaParameters(c_real=-0.4, c_imag=0.6),
    'dendrite': JuliaParameters(c_real=-0.235125, c_imag=0.827215),
    'lightning': JuliaParameters(c_real=-0.8, c_imag=0.156),
    'rabbit': JuliaParameters(c_real=-0.123, c_imag=0.745),
    'airplane': JuliaParameters(c_real=-1.25, c_imag=0.0),
    'san_marco': JuliaParameters(c_real=-0.75, c_imag=0.0),
    'siegel_disk': JuliaParameters(c_real=-0.391, c_imag=-0.587),
}
