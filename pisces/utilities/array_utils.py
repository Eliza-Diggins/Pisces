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
        raise ValueError(f"Expected {ndim_present} axes in `coordinates` based on `axis_mask`, but found {coordinates.shape[-1]}.")
    if fill_values.size != (ndim - ndim_present):
        raise ValueError(f"`fill_values` had size {fill_values.size}, expected {ndim - ndim_present} based on `axis_mask`.")

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
        raise ValueError(f"`grid_axis_mask` length ({len(grid_axis_mask)}) must match the number of dimensions in the final coordinate shape ({ndim}).")
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