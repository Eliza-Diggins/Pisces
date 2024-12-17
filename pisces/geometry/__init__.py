"""
Geometry handling tool for Pisces computations.

The :py:mod:`pisces.geometry` module provides functions for handling geometry in Pisces. This includes a novel system for complex
differential operators in general coordinate systems, as well as other geometry related operations which underlie much of
Pisces' code.

For an overview of the relevant theory, see :ref:`geometry_theory`.

"""
from pisces.geometry.base import CoordinateSystem
from pisces.geometry.coordinate_systems import (
    SphericalCoordinateSystem,
    CartesianCoordinateSystem,
    CylindricalCoordinateSystem,
    PolarCoordinateSystem,
    ProlateHomoeoidalCoordinateSystem,
    ProlateSpheroidalCoordinateSystem,
    OblateHomoeoidalCoordinateSystem,
    OblateSpheroidalCoordinateSystem,
    CartesianCoordinateSystem1D,
    CartesianCoordinateSystem2D,
    PseudoSphericalCoordinateSystem
)
from pisces.geometry.handler import GeometryHandler

__all__ = [
    'CoordinateSystem',
    'SphericalCoordinateSystem',
    'CartesianCoordinateSystem',
    'CylindricalCoordinateSystem',
    'PolarCoordinateSystem',
    'GeometryHandler',
]