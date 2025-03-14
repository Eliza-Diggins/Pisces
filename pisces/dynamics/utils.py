"""
Utility functions for galactic dynamics calculations
"""
from typing import Literal, Union

import numpy as np
from unyt import Unit, unyt_array, unyt_quantity


def relative_potential(
    gravitational_potential: Union[unyt_array, np.ndarray],
    boundary_value: Union[unyt_quantity, float] = 0,
    order: Literal["forward", "backward"] = "forward",
    base_units: Union[str, Unit] = "cm**2/s**2",
):
    r"""
    Compute the relative potential for a provided 1D gravitational potential.

    The relative potential is

    .. math::

        \Psi = -(\Phi - \Phi_0),

    where :math:`\Phi` is the gravitational potential and :math:`\Phi_0` is the boundary value
    of the potential.


    Parameters
    ----------
    gravitational_potential: np.ndarray or unyt_array
        The gravitational potential :math:`\Phi`. If units are specified, then everything is consistently converted to
        base units (specified by the ``base_units`` kwarg). By default, the base units are :math:`{\rm cm^2\; s^{-2}}`. If
        units are not provided, then it is assumed that the potential is already in the correct base units.
    boundary_value: unyt_quantity or float, optional
        The boundary value of the gravitational potential. This can (in principle) be arbitrary; however, it is important
        that it is consistently applied. By default, the boundary value is 0.
    order: str, optional
        The array order to return the relative potential. If the gravitational potential is provided in ascending order, then
        the ``'forward'`` order will lead to a relative potential which is in descending order. Setting ``order='backward'`` will
        instead flip the array so that the array is in ascending order.

        .. hint::

            This can be useful when using the relative potential as the abscissa for an interpolation method.

    base_units: str or Unit, optional
        The base units of the potential and the units in which the relative potential is returned. By default, the base units are :math:`{\rm cm^2\; s^{-2}}`.

    Returns
    -------
    unyt_array
        The relative potential :math:`\Phi`
    """
    _base_units = base_units or "cm**2/s**2"

    # Validate the inputs and ensure that the conversion occurs for the input potential.
    # Convert the potential and then enforce the array behavior. Check dimensionality.
    if hasattr(gravitational_potential, "units"):
        gravitational_potential = gravitational_potential.to_value(_base_units)

    gravitational_potential = np.asarray(gravitational_potential)

    if gravitational_potential.ndim != 1:
        raise ValueError(
            "`relative_potential` only works for 1D gravitational potentials."
        )

    # Convert the boundary value to base units.
    if hasattr(boundary_value, "units"):
        boundary_value = boundary_value.to_value(_base_units)

    # Construct the relative potential
    rpot = -(gravitational_potential - boundary_value)

    if order == "forward":
        return unyt_array(rpot, _base_units)
    else:
        return unyt_array(rpot[::-1], _base_units)


def relative_energy(
    energy: Union[unyt_array, np.ndarray],
    boundary_value: Union[unyt_quantity, float] = 0,
    order: Literal["forward", "backward"] = "forward",
    base_units: Union[str, Unit] = "cm**2/s**2",
):
    r"""
    Compute the relative energy for a provided 1D energy array.

    The relative potential is

    .. math::

        \mathcal{E} = -(\mathcal{E} - \Phi_0),

    where :math:`\mathcal{E}` is the relative energy and :math:`\Phi_0` is the boundary value
    of the potential.


    Parameters
    ----------
    energy: np.ndarray or unyt_array
        The energy array. If units are specified, then everything is consistently converted to
        base units (specified by the ``base_units`` kwarg). By default, the base units are :math:`{\rm cm^2\; s^{-2}}`. If
        units are not provided, then it is assumed that the energy is already in the correct base units.
    boundary_value: unyt_quantity or float, optional
        The boundary value of the gravitational potential. This can (in principle) be arbitrary; however, it is important
        that it is consistently applied. By default, the boundary value is 0.
    order: str, optional
        The array order to return the relative potential. If the energy is provided in ascending order, then
        the ``'forward'`` order will lead to a relative energy which is in descending order. Setting ``order='backward'`` will
        instead flip the array so that the array is in ascending order.

        .. hint::

            This can be useful when using the relative energy as the abscissa for an interpolation method.

    base_units: str or Unit, optional
        The base units of the energy and the units in which the relative energy is returned. By default, the base units are :math:`{\rm cm^2\; s^{-2}}`.

    Returns
    -------
    unyt_array
        The relative energy :math:`\mathcal{E}`
    """
    _base_units = base_units or "cm**2/s**2"

    # Validate the inputs and ensure that the conversion occurs for the input potential.
    # Convert the potential and then enforce the array behavior. Check dimensionality.
    if hasattr(energy, "units"):
        energy = energy.to_value(_base_units)

    energy = np.asarray(energy)

    if energy.ndim != 1:
        raise ValueError(
            "`relative_potential` only works for 1D gravitational potentials."
        )

    # Convert the boundary value to base units.
    if hasattr(boundary_value, "units"):
        boundary_value = boundary_value.to_value(_base_units)

    # Construct the relative potential
    reng = -(energy - boundary_value)

    if order == "forward":
        return unyt_array(reng, _base_units)
    else:
        return unyt_array(reng[::-1], _base_units)
