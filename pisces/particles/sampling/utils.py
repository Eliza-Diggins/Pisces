"""
Utility functions for sampling procedures.
"""
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from pisces.models.base import Model


def compute_likelihood_from_density(
    model: "Model", field_name: str
) -> Tuple[str, np.ndarray]:
    r"""
    Convert a mass density field into a likelihood field using the Jacobian.

    Parameters
    ----------
    model: :py:class:`~pisces.models.base.Model`
        The model containing the ``field_name`` that is being converted to a likelihood.
    field_name: str
        The name of the density field to be converted.

    Returns
    -------
    List of str
        The axes represented in the likelihood field.
    np.ndarray
        The resulting likelihood field as an array.

    Notes
    -----
    Given a density function :math:`\rho({\bf x})`, the probability of finding a particle in a region
    :math:`V` is the ratio of the mass in :math:`V` to the total mass. Thus

    .. math::

        P(V) = \frac{1}{Z} \int_V \rho({\bf x}) dV =  \frac{1}{Z} \int_V \rho({\bf x})J({\bf x}) d^n x,

    where :math:`Z` is the normalization. As such, the likelihood field is

    .. math::

        \mathcal{L} \sim \rho({\bf x})J({\bf x}).

    """
    # Validate that the field exists and extract its axes.
    if field_name not in model.FIELDS:
        raise ValueError(f"No field {field_name} in {model}.")

    _field = model.FIELDS[field_name]
    _field_symbols = set(_field.AXES)

    # Determine the relevant axes from the coordinate system. To start, we need to
    # extract the Jacobian's dependencies and then merge them with that of the field itself.
    _jacobian_free_symbols = model.coordinate_system.get_derived_attribute_symbolic(
        "jacobian"
    ).free_symbols
    _jacobian_free_symbols = {str(_s) for _s in list(_jacobian_free_symbols)}

    _symbols = _jacobian_free_symbols.union(_field_symbols)
    _symbols = model.coordinate_system.ensure_axis_order(list(_symbols))
    _axes_mask = model.coordinate_system.build_axes_mask(_symbols)

    # Evaluate the jacobian on the full set of coordinates for the system, then cut it down
    # to only the relevant axes.
    unscaled_coordinates = model.grid_manager.get_coordinates(
        scaled=False
    )  # Pull all for Jac eval.
    jacobian = model.coordinate_system.jacobian(unscaled_coordinates)
    jacobian = jacobian[
        tuple(
            slice(None) if _a in _symbols else -1 for _a in model.coordinate_system.AXES
        )
    ]

    # Coerce the field and the jacobian to be of the same shape
    density, jacobian = model.grid_manager.make_fields_consistent(
        [_field[...].d, jacobian], [list(_field_symbols), list(_jacobian_free_symbols)]
    )

    # Construct the likelihood as the product
    unscaled_coordinates = model.grid_manager.get_coordinates(
        axes=_symbols, scaled=False
    )  # Pull all for Jac eval.
    likelihood = density * jacobian
    likelihood *= np.prod(
        unscaled_coordinates[..., model.grid_manager.is_log_mask[_axes_mask]], axis=-1
    )

    return _symbols, likelihood
