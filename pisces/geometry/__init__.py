"""
Geometry handling tool for Pisces computations.

The :py:mod:`geometry` module provides functions for handling geometry in Pisces. This includes a novel system for complex
differential operators in general coordinate systems, as well as other geometry related operations which underlie much of
Pisces' code.

"""
from pisces.geometry.abc import CoordinateSystem
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem,CartesianCoordinateSystem,CylindricalCoordinateSystem,PolarCoordinateSystem
from pisces.geometry.symmetry import Symmetry
from pisces.geometry.handlers import GeometryHandler

__all__ = [
    'CoordinateSystem',
    'SphericalCoordinateSystem',
    'CartesianCoordinateSystem',
    'CylindricalCoordinateSystem',
    'PolarCoordinateSystem',
    'Symmetry',
    'GeometryHandler',
]