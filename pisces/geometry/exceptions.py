"""Error classes for the :py:mod:`~pisces.geometry` module."""


class GeometryError(Exception):
    r"""Base exception class for geometry-related errors."""

    pass


class CoordinateTypeError(GeometryError):
    r"""Exception raised when an invalid coordinate type is encountered."""

    pass


class ConversionError(GeometryError):
    r"""Exception raised during failed coordinate conversion operations."""

    pass


class LameCoefficientError(GeometryError):
    r"""Exception raised for errors in Lame coefficient calculations or dependencies."""

    pass


class InvalidSymmetryError(GeometryError):
    r"""Exception raised when an invalid symmetry configuration is detected."""

    pass
