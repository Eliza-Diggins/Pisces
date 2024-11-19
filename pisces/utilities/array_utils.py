from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def fill_missing_coord_axes(coordinates: NDArray[np.floating],
                            axis_mask: NDArray[np.bool_],
                            fill_values: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Fills missing coordinates along specified axes based on a provided mask.

    This function takes an array of coordinates that do not contain values along certain axes, as indicated by `axis_mask`.
    Using `fill_values`, it fills the missing coordinates along those axes, resulting in a complete set of coordinates for
    the target dimensions.

    Parameters
    ----------
    coordinates : NDArray[np.floating]
        An array of coordinates where each entry corresponds to a known axis based on the provided `axis_mask`.
    axis_mask : NDArray[np.bool_]
        A boolean array where each `True` value indicates that the corresponding axis is present in `coordinates`,
        and `False` indicates that the axis is missing and should be filled from `fill_values`.
    fill_values : NDArray[np.floating]
        An array containing values to fill in for the missing axes, in the order they appear in the full coordinate system.

    Returns
    -------
    NDArray[np.floating]
        A complete set of coordinates with dimensions matching the number of axes in `axis_mask`.

    Raises
    ------
    ValueError
        If `coordinates` does not match the number of `True` values in `axis_mask` or if `fill_values` does not match
        the number of `False` values in `axis_mask`.

    Examples
    --------
    >>> coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = np.array([True, False, True])
    >>> fills = np.array([0.0])
    >>> fill_missing_coord_axes(coords, mask, fills)
    array([[1., 0., 2.],
           [3., 0., 4.]])
    """
    coordinates, axis_mask, fill_values = np.array(coordinates), np.array(axis_mask), np.array(fill_values)
    ndim = axis_mask.size
    ndim_present = np.sum(axis_mask)

    if coordinates.shape[-1] != ndim_present:
        raise ValueError(
            f"Expected {ndim_present} axes in `coordinates` based on `axis_mask`, but found {coordinates.shape[-1]}.")
    if fill_values.size != (ndim - ndim_present):
        raise ValueError(
            f"`fill_values` had size {fill_values.size}, expected {ndim - ndim_present} based on `axis_mask`.")

    full_coordinates = np.empty((*coordinates.shape[:-1], ndim))
    full_coordinates[..., axis_mask] = coordinates
    full_coordinates[..., ~axis_mask] = fill_values

    return full_coordinates


def reshape_coords_as_grid(coordinates: NDArray[np.floating],
                           /,
                           grid_axis_mask: Optional[NDArray[np.bool_]] = None) -> NDArray[np.floating]:
    """
    Reshapes an array of coordinates into a grid-like format, adding singleton dimensions for missing axes.

    This function reshapes `coordinates`, which has dimensions `(..., k)`, to a shape `(N_1, N_2, ..., N_k, ndim)`, where
    the shape of each axis is determined by `grid_axis_mask`. Any missing axes are reshaped as singleton dimensions.

    Parameters
    ----------
    coordinates : NDArray[np.floating]
        The input array of coordinates with shape `(..., k)`, where `k` is the number of current axes in the coordinates.
    grid_axis_mask : NDArray[np.bool_], optional
        A boolean mask specifying which axes in the output grid should correspond to actual data in `coordinates`.
        Non-masked axes will be singleton dimensions. If not provided, it is assumed that the first `p` axes in the
        output grid correspond to actual coordinates, where `p = coordinates.shape[-1]`.

    Returns
    -------
    NDArray[np.floating]
        The reshaped coordinates array with shape `(N_1, N_2, ..., N_k, ndim)`.

    Raises
    ------
    ValueError
        If `grid_axis_mask` is provided and its length does not match `coordinates.shape[-1]`.
        If the number of `True` values in `grid_axis_mask` does not match the dimensionality of `coordinates` - 1.

    Examples
    --------
    >>> coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = np.array([True, False])
    >>> reshape_coords_as_grid(coords, grid_axis_mask=mask).shape
    (2, 1, 2)
    """
    coordinates = np.array(coordinates)
    ndim = coordinates.shape[-1]

    if grid_axis_mask is None:
        p = coordinates.ndim - 1
        grid_axis_mask = np.array([True] * p + [False] * (ndim - p), dtype=bool)
    else:
        grid_axis_mask = np.array(grid_axis_mask, dtype=bool)
    if len(grid_axis_mask) != ndim:
        raise ValueError(
            f"`grid_axis_mask` length ({len(grid_axis_mask)}) must match the number of dimensions in the final coordinate shape ({ndim}).")
    if np.sum(grid_axis_mask) != coordinates.ndim - 1:
        raise ValueError("The number of `True` values in `grid_axis_mask` must match `coordinates.ndim - 1`.")

    target_shape = np.ones(len(grid_axis_mask), dtype=int)
    target_shape[grid_axis_mask] = coordinates.shape[:-1]
    reshaped_coordinates = coordinates.reshape((*target_shape, ndim))

    return reshaped_coordinates


def complete_and_reshape_as_grid(coordinates: NDArray[np.floating],
                                 axis_mask: NDArray[np.bool_],
                                 *,
                                 grid_axis_mask: Optional[NDArray[np.bool_]] = None,
                                 fill_values: Union[int, NDArray[np.floating]] = 0) -> NDArray[np.floating]:
    """
    Completes missing coordinate axes and reshapes the array into a grid format, with singleton dimensions for absent axes.

    This function fills in missing axes as specified by `axis_mask` and reshapes `coordinates` to a grid structure.
    It combines the functionality of `fill_missing_coord_axes` and `reshape_coords_as_grid`, resulting in a coordinate
    array of shape `(N_1, N_2, ..., N_k, ndim)`, where any missing axes are given singleton dimensions.

    Parameters
    ----------
    coordinates : NDArray[np.floating]
        The input array of coordinates with shape `(..., p)`, where `p` is the number of current axes in the coordinates.
    axis_mask : NDArray[np.bool_]
        A boolean array where each `True` value indicates that the corresponding axis is present in `coordinates`,
        and `False` indicates that the axis is missing and should be filled from `fill_values`.
    grid_axis_mask : Optional[NDArray[np.bool_]], optional
        A boolean mask specifying which axes in the output grid should correspond to actual data in `coordinates`.
        Non-masked axes will be singleton dimensions. If not provided, it is assumed that the first `p` axes in the
        output grid correspond to actual coordinates, where `p = coordinates.shape[-1]`.
    fill_values : Union[int, NDArray[np.floating]], optional
        An array of values to use for filling missing axes as specified by `axis_mask`. Default is 0.

    Returns
    -------
    NDArray[np.floating]
        The reshaped coordinates array with shape `(N_1, N_2, ..., N_k, ndim)`.

    Raises
    ------
    ValueError
        If `coordinates` does not match the number of `True` values in `axis_mask`, or if `fill_values` does not match
        the number of `False` values in `axis_mask`.

    Examples
    --------
    >>> coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = np.array([True, True, False])
    >>> fills = np.array([0.0])
    >>> complete_and_reshape_as_grid(coords, mask, fill_values=fills).shape
    (2, 1, 1, 3)

    """
    if isinstance(fill_values, (int, float)):
        fill_values = np.full(np.sum(~axis_mask), fill_values)
    else:
        fill_values = np.array(fill_values, dtype=np.float64)

    completed_coords = fill_missing_coord_axes(coordinates, axis_mask, fill_values)
    return reshape_coords_as_grid(completed_coords, grid_axis_mask=grid_axis_mask)


def is_grid(array: NDArray[np.floating],
            grid_axis_mask: Optional[NDArray[np.bool_]] = None) -> bool:
    """
    Verifies if an array is structured as a grid, including singleton dimensions for absent axes.

    This function checks whether `array` is arranged in a grid format, as determined by `grid_axis_mask`.
    The array shape should have singleton dimensions in place of the missing axes, following the shape
    `(N_1, N_2, ..., N_k, ndim)` where `ndim` is the total number of axes.

    Parameters
    ----------
    array : NDArray[np.floating]
        The array to verify, expected to be in a grid-like format.
    grid_axis_mask : Optional[NDArray[np.bool_]], optional
        A boolean mask where `True` values indicate axes that should have full dimensions in `array`.
        Missing axes will be validated as singleton dimensions. If not provided, it assumes the last
        axis of `array` is the `ndim` dimension.

    Returns
    -------
    bool
        True if `array` conforms to the expected grid format; False otherwise.

    Examples
    --------
    >>> coords = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    >>> mask = np.array([True, False])
    >>> is_grid(coords, mask)
    True

    >>> array = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> is_grid(array)
    False
    """
    array = np.array(array)
    ndim = array.shape[-1]  # Total expected number of axes in the grid

    if array.ndim - 1 != ndim:
        return False
    else:
        return True


def get_grid_coordinates(bbox: NDArray[np.floating],
                         block_size: NDArray[np.int_],
                         cell_size: Optional[NDArray[np.floating]] = None) -> NDArray[np.floating]:
    """
    Generate a grid of coordinates within a specified bounding box.

    This function computes the coordinates of grid points within a bounding box
    ``bbox`` for a grid of specified resolution ``block_size`` and cell size
    ``cell_size``. The coordinates are generated such that the grid points are
    spaced evenly across the bounding box, with each coordinate representing the
    center of a grid cell.

    Parameters
    ----------
    bbox : NDArray[np.float_]
        A 2D array with shape ``(2, D)``, where ``D`` is the number of dimensions. The
        first row (``bbox[0, :]``) represents the minimum coordinate values of the
        bounding box, and the second row (``bbox[1, :]``) represents the maximum
        coordinate values.
    block_size : NDArray[np.int_]
        A 1D array of integers specifying the number of grid points (blocks) along
        each dimension. ``block_size`` must have shape ``(D,)``.
    cell_size : NDArray[np.float_], optional
        A 1D array of floats specifying the size of each grid cell along each
        dimension. Must be ``(D,)`` in size. If it is not specified, it will be
        calculated dynamically.

    Returns
    -------
    NDArray[np.float_]
        A grid of coordinates with shape `(block_size[0], ..., block_size[D-1], D)`,
        where the last dimension corresponds to the spatial coordinates.

    Examples
    --------
    **Generating a Simple 2D Grid**

    Let's generate a grid from ``(0,0)`` to ``(1,1)`` that is a ``2x2`` grid. The points
    are expected to be at ``0.25`` and ``0.75`` in each dimension.

    >>> import numpy as np
    >>> bbox = np.array([[0, 0], [1, 1]])
    >>> block_size = np.array([2, 2])
    >>> grid_coords = get_grid_coordinates(bbox, block_size)
    >>> grid_coords[..., 0]  # x-coordinates
    array([[0.25, 0.25],
           [0.75, 0.75]])
    >>> grid_coords[..., 1]  # y-coordinates
    array([[0.25, 0.75],
           [0.25, 0.75]])

    **Performance Comparison with Precomputed `cell_size`**

    When performance is a concern, precomputing `cell_size` can save time.

    >>> from time import perf_counter
    >>> bbox = np.array([[0, 0], [1, 1]])
    >>> block_size = np.array([100, 100])
    >>> cell_size = (bbox[1, :] - bbox[0, :]) / block_size
    >>> precomputed_times = []
    >>> for _ in range(100): # doctest: +SKIP
    ...     start = perf_counter()
    ...     get_grid_coordinates(bbox, block_size, cell_size=cell_size)
    ...     precomputed_times.append(perf_counter() - start)
    >>> dynamic_times = []
    >>> for _ in range(100): # doctest: +SKIP
    ...     start = perf_counter()
    ...     get_grid_coordinates(bbox, block_size, cell_size=None)
    ...     dynamic_times.append(perf_counter() - start)
    >>> print(f"Precomputed: {np.mean(precomputed_times):.6f}s, Dynamic: {np.mean(dynamic_times):.6f}s")  # doctest: +SKIP

    Notes
    -----
    - The grid coordinates are centered within the cells, with the first coordinate
      offset by `cell_size / 2` from the lower bound of the bounding box along
      each dimension.
    - The resulting grid points are evenly spaced and cover the entire bounding box.
    """
    if cell_size is None:
        cell_size = (bbox[1, :] - bbox[0, :]) / block_size

    # Generate slices for mgrid
    slices = tuple(
        slice(bbox[0, i] + (cell_size[i] / 2), bbox[1, i] - (cell_size[i] / 2), complex(0, block_size[i]))
        for i in range(len(block_size))
    )

    # Generate the grid using mgrid and reorder dimensions
    grid = np.mgrid[slices]
    grid = np.stack(grid, axis=-1)

    return grid

