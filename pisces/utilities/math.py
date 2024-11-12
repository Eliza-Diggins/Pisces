import numpy as np
from typing import List, Any, Union

def compute_grid_spacing(coordinates: np.ndarray, axes: Union[List[int], None] = None) -> List[np.ndarray]:
    """
    Extracts 1D coordinate arrays along each specified axis from a coordinate grid.

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinate grid of shape ``(*grid_shape, NDIM)``, where ``grid_shape`` is the shape of the underlying
        grid and ``NDIM`` is the number of dimensions. Each index of the final axis corresponds to a given coordinate
        in the coordinate system.

        .. hint::

            On an ``(N_1,...,N_k)`` grid, the coordinates should be ``(N_1,...,N_k,k)`` in shape. You can use
            ``np.stack([coords],axis=-1)`` to construct this given ``k`` coordinate arrays.

    axes : List[int], optional
        The axes along which to extract coordinates. If ``None`` (default), all axes are extracted.

    Returns
    -------
    List[np.ndarray]
        A list of 1D coordinate arrays along each of the axes in ``axes``. For the ``k``-th axis, the returned array should
        be ``(N_k,)`` in shape and contain the ``k``-th coordinate value for each slice of the grid perpendicular to the
        ``k``-th coordinate direction.
    """
    # Compute the number of dimensions in the space and coerce the axes so that
    # they are present regardless of input.
    ndim = coordinates.shape[-1]
    if axes is None:
        axes = list(range(ndim))

    # Slice through the correct coordinate array.
    try:
        return [
            coordinates[(0,)*axis + (slice(None),) + (ndim-axis-1)*(0,) + (axis,)]
            for axis in axes
        ]
    except Exception as e:
        raise ValueError(f"Failed to compute grid spacing: {e}")


def partial_derivative(coordinates: np.ndarray,
                       field: np.ndarray,
                       /,
                       spacing: Any = None,
                       *,
                       axes: Union[List[int], int, None] = None,
                       __validate__: bool = True,
                       **kwargs) -> np.ndarray:
    """
    Compute partial derivatives of a field over specified axes, using custom spacing if provided.

    Parameters
    ----------
    coordinates : np.ndarray
        Coordinate grid of shape ``(*grid_shape, NDIM)``, where ``grid_shape`` is the shape of the underlying
        grid and ``NDIM`` is the number of dimensions. Each index of the final axis corresponds to a given coordinate
        in the coordinate system.

        .. hint::

            On an ``(N_1,...,N_k)`` grid, the coordinates should be ``(N_1,...,N_k,k)`` in shape. You can use
            ``np.stack([coords],axis=-1)`` to construct this given ``k`` coordinate arrays.

    field : np.ndarray
        The field values on which to compute the derivative. These should be of shape ``(*grid_shape,)`` with each element
        representing the value at the grid position specified by the indices.
    spacing : Any, optional
        Spacing between ``field`` values. By default, this will be computed from the
        provided coordinates.

        Spacing can be specified using:

        1. single scalar to specify a sample distance for all dimensions.
        2. N scalars to specify a constant sample distance for each dimension.
           i.e. `dx`, `dy`, `dz`, ...
        3. N arrays to specify the coordinates of the values along each
           dimension of F. The length of the array must match the size of
           the corresponding dimension
        4. Any combination of N scalars/arrays with the meaning of 2. and 3.

        If `axis` is given, the number of varargs must equal the number of axes.
        Default: 1.
    axes : list of int or int, optional
        The axis or list of axes along which to compute the derivative. If None, derivatives
        will be computed along all axes.
    __validate__ : bool, optional
        Whether to validate `coordinates` and `field` shapes for consistency.
    **kwargs : Additional arguments passed to `np.gradient`.

    Returns
    -------
    np.ndarray
        Array of the computed partial derivative(s) of `field`. Will have shape ``(*grid_shape, axes)`` with each
        element of the final axis representing a partial derivative along a different axis.

    Raises
    ------
    ValueError
        If `coordinates` shape is incompatible with the required grid format or if validation
        fails.
    """
    # Validation
    if __validate__:
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 1)

        # Verify that coordinates shape matches grid dimensions plus spatial dimension
        if coordinates.shape[-1] != coordinates.ndim - 1:
            raise ValueError(
                f"Coordinates have shape {coordinates.shape}, which is invalid because "
                f"the final axis indicates a spatial dimension of {coordinates.shape[-1]}, "
                f"but only {coordinates.ndim - 1} grid axes are present."
            )

    # Compute spacing if none provided
    if spacing is None:
        spacing = compute_grid_spacing(coordinates, axes=axes)

    # Handle np.gradient call for partial derivatives along specified axes
    return np.gradient(field, *spacing, axis=axes, **kwargs)
