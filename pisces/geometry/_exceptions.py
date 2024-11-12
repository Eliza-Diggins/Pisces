class GeometryError(Exception):
    """Base exception class for geometry-related errors."""
    pass

class CoordinateTypeError(GeometryError):
    """Exception raised when an invalid coordinate type is encountered."""
    pass

class ConversionError(GeometryError):
    """Exception raised during failed coordinate conversion operations."""
    pass

class LameCoefficientError(GeometryError):
    """Exception raised for errors in Lame coefficient calculations or dependencies."""
    pass

class InvalidSymmetryError(GeometryError):
    """Exception raised when an invalid symmetry configuration is detected."""
    pass