"""
Array manipulations module for Pisces.

This module provides classes and utility functions for managing coordinate arrays and grids,
ensuring proper dimensional consistency and grid formatting.

Array Requirements
-------------------

- Coordinate Arrays: Shape ``(..., N)``, where ``N`` is the number of coordinate axes currently represented.
- Grids: Shape ``(N_1, N_2, ..., N_NDIM, NDIM)``, where ``NDIM`` is the total number of axes in the grid.

"""
import warnings
from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray

from pisces.utilities.logging import devlog


# @@ COORDINATE MANAGEMENT @@ #
# These classes and utility functions are all focused on ensuring proper formatting
# for coordinate grids and coordinate sets.
#
# DEVELOPERS: remember that a coordinate set can be any (...,NDIM) array, but a grid must have
#   shape (N_1,N_2,...,N_NDIM,NDIM) to be valid.
# noinspection PyUnresolvedReferences
class CoordinateArray(np.ndarray):
    """
    A subclass of numpy.ndarray that enforces dimensional consistency for coordinate arrays.
    This class validates and reshapes the array to ensure compatibility with an ``NDIM`` coordinate system.

    Parameters
    ----------
    input_array : array-like
        The array data to be stored, expected to represent coordinates. This should be ``(...,NDIM)`` in shape,
        where ``NDIM`` is the total number of dimensions.
    ndim : int, optional
        The number of dimensions for the coordinate system. If not provided,
        it is inferred based on the shape of the input array.

    Notes
    -----
    - If the input array is 1D, it is interpreted as a single coordinate unless ``ndim`` is provided.
    - If the input array shape is incompatible with the given or inferred ``ndim``, an error is raised.
    - This class outputs a regular ndarray when numpy operations are performed on it.
    """

    def __new__(cls, input_array, ndim=None):
        # CONVERT to np array to avoid any type issues.
        obj = np.asarray(input_array).view(cls)

        # Validate or infer ndim
        if ndim is None:
            if obj.ndim == 1:
                ndim = obj.shape[0]
            elif obj.ndim > 1:
                ndim = obj.shape[-1]
        else:
            ndim = int(ndim)

        # Validate the shape of the array
        if (obj.ndim == 1) and (ndim == 1):
            # We've been given an array of points in a 1D coord space, -> (N,1)
            devlog.debug(
                "Reshaping coordinate array: %s -> %s.", obj.shape, (obj.size, 1)
            )
            obj = obj.reshape((obj.size, 1))
        elif (obj.ndim == 1) and (ndim > 1):
            # We've been given a single point in an ND space. -> (1,N)
            if obj.size != ndim:
                raise ValueError(
                    f"COORDINATE COERCION FAILED: got 1D array of length {obj.size}, but ndim={ndim} which is neither 1 or {obj.size}."
                )
            devlog.debug(
                "Reshaping coordinate array: %s -> %s.", obj.shape, (1, obj.size)
            )
            obj = obj.reshape((1, obj.size))
        elif obj.ndim > 1:
            # We have a multidimensional object. We check the final dimension and the first dimension before
            # moving forward.
            if obj.shape[-1] == ndim:
                return obj

            # check if the first dimension matches
            if obj.shape[0] == ndim:
                warnings.warn(
                    f"COORDINATE COERCION WARNING: got a {obj.shape} array, which seems to have the NDIM axis in position 0 instead"
                    " of at the end. This is corrected here, but is unsafe and should be fixed."
                )
                devlog.debug("Reshaping coordinate array: moving axis 0 -> -1")
                return np.moveaxis(obj, 0, -1)

            raise ValueError(
                f"COORDINATE COERCION FAILED: Input array's first or last dimension must match ndim={ndim}. Got {obj.shape[-1]} instead."
            )
        else:
            raise ValueError(
                "COORDINATE COERCION FAILED: Input array must be at least 1D."
            )

        # Return the validated and reshaped object
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array view, ensuring operations result in regular np.ndarray.

        Parameters
        ----------
        obj : ndarray or None
            The original array from which the new object was created.
        """
        if obj is None:
            return  # Called during explicit construction, no additional setup needed

        # Convert any sliced or operated result back to a plain np.ndarray
        if type(self) is not CoordinateArray:
            return np.asarray(self)


# noinspection PyUnresolvedReferences
class CoordinateGrid(CoordinateArray):
    """
    A subclass of :py:class:`CoordinateArray` that enforces the grid standard.

    This class ensures the array has the correct grid shape ``(N_1, N_2, ..., N_NDIM, NDIM)``.
    Missing axes can be filled in based on the ``complete_axes`` and ``present_axes`` parameters.

    Parameters
    ----------
    input_array : array-like
        The input array of coordinates. Should be ``(N_1,...,N_M, M)`` where ``M`` is the number of currently represented axes.
    ndim : int, optional
        The number of dimensions for the coordinate system. If not provided,
        it is inferred based on the shape of the input array.
    axes_mask: np.ndarray, optional
        Boolean array indicating which axes are currently present in the ``input_array``.
        ``True`` values correspond to present axes, and ``False`` values correspond to missing axes.

    Notes
    -----
    - If ``complete_axes`` and ``present_axes`` are provided, the grid is reshaped to include missing axes as singleton dimensions.
    - Axes not present in the ``present_axes`` are filled with singleton dimensions.
    """

    def __new__(cls, input_array, ndim=None, axes_mask=None):
        # Initialize the superclass. This should enforce the standard for the coordinate array.
        # This leaves this class with only needing to check for the grid standard.
        obj = super().__new__(cls, input_array, ndim=ndim)

        # Construct a set of axes and complete axes regardless.
        ndim = obj.shape[
            -1
        ]  # Needed because super().__new__ fills this in, but we don't have access.
        if axes_mask is None:
            devlog.warning(
                "CoordinateGrid instance missing axes specification -> grid check may be insufficient."
            )
            axes_mask = np.array(
                [True if i < len(obj.shape) - 1 else False for i in np.arange(ndim)],
                dtype=bool,
            )

        devlog.warning(axes_mask)
        # Validate that present axes has the right length.
        if np.sum(axes_mask) != (obj.ndim - 1):
            raise ValueError(
                f"CoordinateGrid got `axes_mask`:{axes_mask} input array had {obj.ndim - 1} free axes."
            )

        if len(axes_mask) != ndim:
            raise ValueError(
                f"CoordinateGrid got `axes_mask`:{axes_mask} but `ndim`={ndim} is expected."
            )

        # Now cycle through and fill out the shape.
        if not all(ax for ax in axes_mask):
            _new_shape = np.ones_like(
                axes_mask, dtype=np.uint32
            )  # (ndim,) array of ones.
            _new_shape[axes_mask] = obj.shape[:-1]

            devlog.debug(
                "CoordinateGrid reshaping %s to %s.",
                obj.shape,
                tuple([*_new_shape, ndim]),
            )
            obj = obj.reshape((*_new_shape, ndim))

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return


def fill_missing_coord_axes(
    coordinates: NDArray[np.floating],
    axis_mask: NDArray[np.bool_],
    fill_values: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Fills missing coordinates along specified axes based on a provided mask.

    This function takes an array of coordinates that do not contain values along certain axes, as indicated by ``axis_mask``.
    Using ``fill_values``, it fills the missing coordinates along those axes, resulting in a complete set of coordinates for
    the target dimensions.

    Parameters
    ----------
    coordinates : NDArray[np.floating]
        An array of coordinates where each entry corresponds to a known axis based on the provided ``axis_mask``.

        This should be a generic ``(...,N)`` array where ``N`` is some number of axes present in the ``coordinates`` array.
        The ``axis_mask`` should have ``N`` true values and ``NDIM-N`` false values.
    axis_mask : NDArray[np.bool_]
        A boolean array where each ``True`` value indicates that the corresponding axis is present in ``coordinates``,
        and ``False`` indicates that the axis is missing and should be filled from ``fill_values``. There should be ``N``
        ``True`` and ``NDIM-N`` ``False``.
    fill_values : NDArray[np.floating]
        An array containing values to fill in for the missing axes, in the order they appear in the full coordinate system.
        This should be ``(NDIM-N,)`` in shape.
    Returns
    -------
    NDArray[np.floating]
        A complete set of coordinates with dimensions matching the number of axes in ``axis_mask``. The output shape will
        be ``(..., NDIM)``.

    Raises
    ------
    ValueError
        If ``coordinates`` does not match the number of ``True`` values in ``axis_mask`` or if ``fill_values`` does not match
        the number of ``False`` values in ``axis_mask``.

    Examples
    --------
    >>> coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = np.array([True, False, True])
    >>> fills = np.array([0.0])
    >>> fill_missing_coord_axes(coords, mask, fills)
    array([[1., 0., 2.],
           [3., 0., 4.]])
    """
    coordinates, axis_mask, fill_values = (
        np.array(coordinates),
        np.array(axis_mask),
        np.array(fill_values),
    )
    ndim = axis_mask.size
    ndim_present = np.sum(axis_mask)

    if coordinates.shape[-1] != ndim_present:
        raise ValueError(
            f"Expected {ndim_present} axes in `coordinates` based on `axis_mask`, but found {coordinates.shape[-1]}."
        )
    if fill_values.size != (ndim - ndim_present):
        raise ValueError(
            f"`fill_values` had size {fill_values.size}, expected {ndim - ndim_present} based on `axis_mask`."
        )

    full_coordinates = np.empty((*coordinates.shape[:-1], ndim))
    full_coordinates[..., axis_mask] = coordinates
    full_coordinates[..., ~axis_mask] = fill_values

    return full_coordinates


def complete_and_reshape_as_grid(
    coordinates: NDArray[np.floating],
    axes_mask: NDArray[np.bool_],
    *,
    grid_axes_mask: Optional[NDArray[np.bool_]] = None,
    fill_values: Union[int, NDArray[np.floating]] = 0,
) -> NDArray[np.floating]:
    """
    Completes missing coordinate axes and reshapes the array into a grid format, with singleton dimensions for absent axes.

    This function fills in missing axes as specified by ``axis_mask`` and reshapes ``coordinates`` to a grid structure.
    It combines the functionality of ``fill_missing_coord_axes`` and ``reshape_coords_as_grid``, resulting in a coordinate
    array of shape ``(N_1, N_2, ..., N_k, ndim)``, where any missing axes are given singleton dimensions.

    Parameters
    ----------
    coordinates : NDArray[np.floating]
        The input array of coordinates with shape ``(..., p)``, where ``p`` is the number of current axes in the coordinates.
    axes_mask : NDArray[np.bool_]
        A boolean array where each ``True`` value indicates that the corresponding axis is present in ``coordinates``,
        and ``False`` indicates that the axis is missing and should be filled from ``fill_values``.
    grid_axes_mask : Optional[NDArray[np.bool_]], optional
        A boolean mask specifying which axes in the output grid should correspond to actual data in ``coordinates``.
        Non-masked axes will be singleton dimensions. If not provided, it is assumed that the first ``p`` axes in the
        output grid correspond to actual coordinates, where ``p = coordinates.shape[-1]``.
    fill_values : Union[int, NDArray[np.floating]], optional
        An array of values to use for filling missing axes as specified by ``axis_mask``. Default is 0.

    Returns
    -------
    NDArray[np.floating]
        The reshaped coordinates array with shape ``(N_1, N_2, ..., N_k, ndim)``.

    Raises
    ------
    ValueError
        If ``coordinates`` does not match the number of ``True`` values in ``axis_mask``, or if ``fill_values`` does not match
        the number of ``False`` values in ``axis_mask``.

    Examples
    --------
    >>> coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = np.array([True, True, False])
    >>> grid_mask = np.array([True, False, False])
    >>> fills = np.array([0.0])
    >>> complete_and_reshape_as_grid(coords, mask,grid_axes_mask=grid_mask, fill_values=fills).shape
    (2, 1, 1, 3)

    """
    if isinstance(fill_values, (int, float)):
        fill_values = np.full(np.sum(~axes_mask), fill_values)
    else:
        fill_values = np.array(fill_values, dtype=np.float64)

    completed_coords = fill_missing_coord_axes(coordinates, axes_mask, fill_values)
    return CoordinateGrid(
        completed_coords, ndim=len(axes_mask), axes_mask=grid_axes_mask
    )


def get_grid_coordinates(
    bbox: NDArray[np.floating],
    block_size: NDArray[np.int_],
    cell_size: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
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
        A grid of coordinates with shape ``(block_size[0], ..., block_size[D-1], D)``,
        where the last dimension corresponds to the spatial coordinates.

    Examples
    --------
    **Generating a Simple 2D Grid**

    Let's generate a grid from ``(0,0)`` to ``(1,1)`` that is a ``2x2`` grid. The points
    are expected to be at ``0.25`` and ``0.75`` in each dimension.

    >>> _bbox = np.array([[0, 0], [1, 1]])
    >>> _block_size = np.array([2, 2])
    >>> grid_coords = get_grid_coordinates(_bbox, _block_size)
    >>> grid_coords[..., 0]  # x-coordinates
    array([[0.25, 0.25],
           [0.75, 0.75]])
    >>> grid_coords[..., 1]  # y-coordinates
    array([[0.25, 0.75],
           [0.25, 0.75]])

    **Performance Comparison with Precomputed ``cell_size``**

    When performance is a concern, precomputing ``cell_size`` can save time.

    >>> from time import perf_counter
    >>> _bbox = np.array([[0, 0], [1, 1]])
    >>> _block_size = np.array([100, 100])
    >>> _cell_size = (_bbox[1, :] - _bbox[0, :]) / _block_size
    >>> precomputed_times = []
    >>> for _ in range(100): # doctest: +SKIP
    ...     start = perf_counter()
    ...     get_grid_coordinates(_bbox, _block_size, cell_size=_cell_size)
    ...     precomputed_times.append(perf_counter() - start)
    >>> dynamic_times = []
    >>> for _ in range(100): # doctest: +SKIP
    ...     start = perf_counter()
    ...     get_grid_coordinates(_bbox, _block_size, cell_size=None)
    ...     dynamic_times.append(perf_counter() - start)
    >>> print(f"Precomputed: {np.mean(precomputed_times):.6f}s, Dynamic: {np.mean(dynamic_times):.6f}s")  # doctest: +SKIP

    Notes
    -----
    - The grid coordinates are centered within the cells, with the first coordinate
      offset by ``cell_size / 2`` from the lower bound of the bounding box along
      each dimension.
    - The resulting grid points are evenly spaced and cover the entire bounding box.
    """
    if cell_size is None:
        cell_size = (bbox[1, :] - bbox[0, :]) / block_size

    # Generate slices for mgrid
    slices = tuple(
        slice(
            bbox[0, i] + (cell_size[i] / 2),
            bbox[1, i] - (cell_size[i] / 2),
            complex(0, block_size[i]),
        )
        for i in range(len(block_size))
    )

    # Generate the grid using mgrid and reorder dimensions
    grid = np.mgrid[slices]
    grid = np.stack(grid, axis=-1)

    return grid


def make_grid_fields_broadcastable(arrays, axes, coordinate_system, field_rank=0):
    """
    Ensures that multiple arrays are broadcastable based on a shared coordinate system.

    This function reshapes input arrays to ensure they are mutually broadcastable according to
    their respective axes and a shared coordinate system. It checks for consistency between
    the arrays, their associated axes, and the canonical axes defined in the coordinate system.

    Parameters
    ----------
    arrays : list[NDArray[np.floating]]
        A list of arrays to reshape for broadcastability. Each array's shape must correspond to
        its specified axes.
    axes : list[list[str]]
        A list of axis specifications for each array. Each element is a list of axis names,
        indicating the dimensions along which the respective array varies.
    coordinate_system : :py:class:`~pisces.geometry.base.CoordinateSystem`
        An object representing the coordinate system, which must have an attribute ``AXES``
        containing the canonical list of valid axis names.
    field_rank : int, optional
        The rank of the field to include in the broadcastability check. Default is 0.

    Returns
    -------
    list[NDArray[np.floating]]
        A list of reshaped arrays, all of which are mutually broadcastable based on the
        canonical axes of the coordinate system.

    Raises
    ------
    ValueError
        If the lengths of ``arrays`` and ``axes`` do not match, if any axis in ``axes`` is not
        present in the canonical ``coordinate_system.AXES``, or if the arrays are not
        mutually broadcastable after reshaping.

    Examples
    --------

    >>> class CoordinateSystem:
    ...     AXES = ['x', 'y', 'z']
    >>> arrs = [np.ones((3, 2)), np.ones((2, 4))]
    >>> _axes = [['x', 'y'], ['y', 'z']]
    >>> reshaped_arrays = make_grid_fields_broadcastable(arrs, _axes, CoordinateSystem())
    >>> [arr.shape for arr in reshaped_arrays]
    [(3, 2, 1), (1, 2, 4)]
    """
    # Validate that arrays and axes are the same length.
    if len(arrays) != len(axes):
        raise ValueError("Arrays and axes must have same length.")

    # Use the coordinate system to extract the canonical axes present.
    # We then proceed to identify the set of all axes present and check
    # that they are all valid axes.
    canonical_axes = coordinate_system.AXES
    present_axes = [ax for ax in canonical_axes if ax in set().union(*axes)]

    if any(ax not in canonical_axes for ax in present_axes):
        raise ValueError(
            f"All axes must be in canonical axes: {canonical_axes}. Got {present_axes}."
        )

    # Now iterate through each of the arrays and axes to
    # reshape into the correct shapes.
    _full_shape = np.ones(len(present_axes), dtype=int)

    for array_index, _axes in enumerate(axes):
        # Create a mask to grab out the elements of the new shape that
        # should have non-unit elements.
        _new_shape = np.ones(len(present_axes), dtype=int)
        _shape_mask = np.array([ax in _axes for ax in present_axes], dtype=bool)
        _new_shape[_shape_mask] = arrays[array_index].shape

        # Reshape the array using this new shape.
        arrays[array_index] = arrays[array_index].reshape(_new_shape)

        # Check that everything is valid with the _full_shape. We either want _full_shape to be
        # 1 or we need the shapes to match. Otherwise broadcastability breaks down.
        if any(
            (ax_fshape != 1) and (ax_shape != 1) and (ax_fshape != ax_shape)
            for ax_shape, ax_fshape in zip(_new_shape, _full_shape)
        ):
            raise ValueError(
                f"Arrays are not mutually broadcastable. {_new_shape} is not broadcastable with {_full_shape}."
            )

        _full_shape = np.array(
            [
                _new_shape[k] if _new_shape[k] != 1 else _full_shape[k]
                for k in range(len(present_axes))
            ]
        )

    return arrays


def build_image_coordinate_array(
    extent: np.ndarray, resolution: np.ndarray, axis: str, position: float
):
    """
    Generates a 3D array of Cartesian coordinates to act as the backing for image generation scripts.

    Parameters
    ----------
    extent: np.ndarray[float]
        A ``(2,2)`` array specifying the lower (``[0,:]``) and upper (``[1,:]``) bounds of the image coordinate
        grid excluding the fixed axis specified by ``axis``.
    resolution: np.ndarray[float]
        A length 2 array specifying the number of pixels along each dimension of the grid.
    axis: str
        One of ``x``,``y``, or ``z``. The axis along which to fix the grid. This axis is perpendicular to
        the generated grid.
    position: float
        The position along which the grid is placed.

    Returns
    -------
    np.ndarray:
        A ``(*resolution, 3)`` array of Cartesian coordinates.
    """
    # Validate inputs: ensure that the resolution, extent, and position are each
    # reasonable values for the arguments.
    extent = np.array(extent, dtype=float)
    resolution = np.array(resolution, dtype=np.uint32)

    if len(extent) != 4:
        raise ValueError(
            f"Argument `extent` should have 4 elements, not {len(extent)}."
        )
    if len(resolution) != 2:
        raise ValueError(
            f"Argument `resolution` should have 2 elements, not {len(resolution)}."
        )

    extent = np.reshape(extent, (2, 2))
    # Determine the fixed axis index from the provided `axis` argument. Raise
    # an error if the axis doesn't exist. For the remaining axes, we need to
    # know the correct indices for them.
    axis_map: Dict[str, int] = dict(x=0, y=1, z=2)
    if axis not in axis_map:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")
    fixed_axis = axis_map[axis]

    # Create the ranges for the two variable axes
    axes_indices = np.array([v for k, v in axis_map.items() if k != axis], dtype=int)

    # Construct the grid ranges and the resulting meshgrid from the
    # specified extent and resolution.
    # NOTE: The resulting meshgrid is (2, *RESOLUTION)
    grid_linspaces = [np.linspace(*extent[i], resolution[i]) for i in range(2)]
    meshgrid = np.stack(np.meshgrid(*grid_linspaces, indexing="ij"), axis=0)

    # reshape the meshgrid to (*RESOLUTION,2 )
    meshgrid = np.moveaxis(meshgrid, 0, -1)

    # Construct the full coordinate set. We use the meshgrid
    # to fill in the relevant axes and then the position to fill the final axis.
    # Create an empty array for the coordinates
    coordinate_array = np.empty((*resolution, 3), dtype=float)
    coordinate_array[..., axes_indices] = meshgrid
    coordinate_array[..., fixed_axis] = position

    return coordinate_array
