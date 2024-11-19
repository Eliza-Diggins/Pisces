import numpy as np
from typing import List, Any, Union, Callable, Literal

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

def function_partial_derivative(
        func: Callable[[np.ndarray],np.ndarray],
        coordinates: np.ndarray,
        axes: Union[int, List[int]],
        method: Literal['forward', 'backward', 'central'] = 'central',
        h: float = 1e-5
) -> np.ndarray:
    """
    Computes numerical partial derivatives of a function at specified coordinates along
    given axes using finite difference methods.

    Parameters
    ----------
    func : Callable
        The function of which to take the partial derivatives. This function must take ``NDIM`` arguments, where
        ``NDIM`` is the number of dimensions of the input arguments.
    coordinates : np.ndarray
        Array of shape ``(N, NDIM)`` where ``N`` is the number of points and ``NDIM`` is
        the number of dimensions. Must match with ``func``'s call signature.
    axes : int or List[int]
        The axis or axes along which to compute the derivative.
    method : {'forward', 'backward', 'central'}, optional
        Finite difference method to use for derivative approximation. Default is 'central'.
    h : float, optional
        Step size for finite differences. Default is 1e-5.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, len(axes))`` representing the partial derivatives at each point
        along each specified axis.

    Examples
    --------

    >>> import numpy as np
    >>> x = np.linspace(-np.pi,np.pi/1000)
    >>> y = np.sin
    >>> dy = function_partial_derivative(y, x,0)
    >>> print(dy)
    """
    # Coerce axes so that we always have an array of axes ints.
    if isinstance(axes, int):
        axes = [axes]
    axes = np.array(axes,dtype='uint32')

    # Ensure that coordinates are a valid structure.
    # We take a transpose here so that we can use *args later.
    if coordinates.ndim == 1:
        coordinates = coordinates.reshape(1,-1)
    else:
        coordinates = coordinates.T

    # Prepare an empty array for the outputs.
    partial_derivatives = np.empty((coordinates.shape[1],axes.size))

    # Perform the differencing method and proceed.
    for i, axis in enumerate(axes):
        # Create a copy of coordinates for modification
        forward_coords = np.copy(coordinates)
        backward_coords = np.copy(coordinates)

        # Adjust coordinates for forward and backward steps
        forward_coords[axis,:] += h
        backward_coords[axis, :] -= h

        if method == 'forward':
            # Forward difference: f(x + h) - f(x) / h
            partial_derivatives[:, i] = (func(*forward_coords) - func(*coordinates)) / h
        elif method == 'backward':
            # Backward difference: f(x) - f(x - h) / h
            partial_derivatives[:, i] = (func(*coordinates) - func(*backward_coords)) / h
        elif method == 'central':
            # Central difference: (f(x + h) - f(x - h)) / (2 * h)
            partial_derivatives[:, i] = (func(*forward_coords) - func(*backward_coords)) / (2 * h)
        else:
            raise ValueError("Unsupported method. Use 'forward', 'backward', or 'central'.")

    return partial_derivatives

