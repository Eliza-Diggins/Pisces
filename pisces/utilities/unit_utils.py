"""
Unit management utility functions.
"""
from numbers import Number

from numpy._typing import ArrayLike
from unyt import Unit, unyt_array, unyt_quantity


def ensure_ytquantity(
    x: Number | unyt_quantity, default_units: Unit | str
) -> unyt_quantity:
    """Ensure that an input ``x`` is a unit-ed quantity with the expected units.

    Parameters
    ----------
    x: Any
        The value to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_quantity
        The corresponding quantity with correct units.
    """
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(x: ArrayLike, default_units: Unit | str) -> unyt_array:
    """Ensure that an input ``x`` is a unit-ed array with the expected units.

    Parameters
    ----------
    x: Any
        The values to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_array
        The corresponding array with correct units.
    """
    if isinstance(x, unyt_array):
        return x.to(default_units)
    elif isinstance(x, tuple) and len(x) == 2:
        return unyt_array(*x).to(default_units)
    else:
        return unyt_array(x, default_units)
