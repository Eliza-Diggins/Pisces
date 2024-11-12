from typing import Callable, Dict, Union
import numpy as np
from numpy.typing import NDArray
from unyt import unyt_array

# Type Aliases for Enhanced Type Checking and Clarity

LameCoefficientFunction = Callable[[NDArray[float]], NDArray[float]]
"""
A callable that accepts an array of coordinates and returns the corresponding Lame coefficient values
as an array. This function maps each point in space to a scaling factor specific to the geometry of
the coordinate system, useful in differential calculations.

Expected Input: NDArray[float] – An array representing points in space (shape depends on dimensionality).
Output: NDArray[float] – Array of Lame coefficient values at those points.
"""

LameCoefficientMap = Dict[int, LameCoefficientFunction]
"""
A dictionary mapping axis indices to their respective Lame coefficient functions.

Each entry corresponds to an axis in an orthogonal coordinate system and maps that axis to its
Lame coefficient function, which defines the scaling factor needed to perform differential
operations along that axis in non-Cartesian geometries.
"""

InvarianceArray = NDArray[np.bool_]
"""
A boolean array indicating axis dependencies for Lame coefficients.

Each element of this array represents whether a particular axis is invariant with respect to
a Lame coefficient. True values indicate invariance along that axis, allowing optimizations in
computations by ignoring constant scaling factors along certain directions.
"""

UnitsPossibleArray = Union[unyt_array, NDArray[float]]
"""
An array type that can either be a `unyt_array` for unit-aware calculations or a standard NumPy array.

Using this type enables handling of physical quantities with units alongside standard arrays,
which is useful for scientific applications that require units. The `unyt_array` type adds
dimensional analysis capabilities, ensuring consistent units in mathematical operations.

Supported Types:
- `unyt_array`: For arrays with units, provided by the `unyt` library.
- `NDArray[float]`: For unitless arrays of floating-point numbers.
"""
