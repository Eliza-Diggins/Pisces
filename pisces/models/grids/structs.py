"""
Type structures for model grid objects.
"""
from typing import Iterable, Union, List

import numpy as np

class BoundingBox(np.ndarray):
    """
    A class representing a bounding box, inheriting from np.ndarray.
    Expected to be a 2xNDIM array where the first row represents
    the minimum bounds and the second row represents the maximum bounds.

    Attributes
    ----------

    Methods
    -------

    """

    def __new__(cls, input_array):
        """
        Create a new BoundingBox instance and validate its shape.

        Parameters
        ----------
        input_array : array-like
            Input data to be cast as a BoundingBox. Must be shape (2, NDIM).

        Returns
        -------
        BoundingBox
            A new BoundingBox instance.
        """
        # Type coercion: must be array cast-able and float64.
        try:
            array = np.asarray(input_array, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to coerce input to BoundingBox: {e}") from e

        # Dimensionality coercion: must be 1D (create a new dim) or 2D.
        if array.ndim == 1:
            array = array[:,np.newaxis]
        elif array.ndim > 2:
            raise ValueError(f"BoundingBox must be 2D (was {array.ndim}D).")

        # Enforce shape requirements
        if array.shape == (array.size//2,2):
            array = array.T

        if array.shape != (2,array.size//2):
            # The shape isn't correct. We want to try to rework it.
            try:
                array = array.reshape((2,array.size//2))
            except Exception as e:
                raise ValueError(f"Could not reshape input array to be a valid BoundingBox: {e}.") from e

        # Validate the values
        if np.any(array[0,:] > array[1,:]):
            raise ValueError(f"BoundingBox cannot have LL corner above UR corner.")

        # Create the DomainDimensions instance as a view of the validated array
        obj = array.view(cls)
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
        if type(self) is not BoundingBox:
            self.__class__ = np.ndarray


class DomainDimensions(np.ndarray):
    """
    Array representation of a grid's domain dimensions.

    This class descents from ``np.ndarray`` and is simply a ``(NDIM,)`` array specifying
    the number of grid cells in each dimension of a grid.

    Attributes
    ----------

    Methods
    -------
    """

    def __new__(cls, input_array):
        """
        Create a new DomainDimensions instance, coercing shape and orientation if needed.

        Parameters
        ----------
        input_array : array-like
            Input data to be cast as DomainDimensions. Should be a 1D array of positive integers
            or a 2D array with shape ``(NDIM, 1)`` or ``(1,NDIM)``. If it is a 1D array, it will be reshaped automatically.

        Returns
        -------
        DomainDimensions
            A new DomainDimensions instance.

        Raises
        ------
        ValueError
            If input cannot be coerced to a 1D array of positive unsigned 32-bit integers.
        """
        # Attempt to coerce input to a uint32 numpy array
        try:
            array = np.asarray(input_array, dtype=np.uint32)
        except Exception as e:
            raise ValueError(f"Failed to coerce input to DomainDimensions: {e}") from e

        # Enforce dimensionality
        if array.ndim != 1:
            raise ValueError(f"DomainDimensions must be 1D (was {array.ndim}D).")

        # Enforce shape requirements
        if array.shape != (array.size,):
            # The shape isn't correct. We want to try to rework it.
            try:
                array = array.reshape((array.size,))
            except Exception as e:
                raise ValueError(f"Could not reshape input array to be a valid set of DomainDimensions: {e}.") from e


        # Create the DomainDimensions instance as a view of the validated array
        obj = array.view(cls)
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
        if type(self) is not DomainDimensions:
            self.__class__ = np.ndarray


class ChunkIndex(np.ndarray):
    """
    A subclass of numpy.ndarray that enforces validation norms for chunk indices in
    :py:class:`~pisces.models.grids.base.ModelGridManager`.

    Parameters
    ----------
    input_array : int or list of int
        The base chunk index. If the grid manager is a 1-D system, then this may be an integer. Otherwise,
        it needs to be some iterable of integers.
    nchunks: np.ndarray[int], optional
        The number of chunks present in each dimension.

    """

    def __new__(cls, index: Union[int, List,np.ndarray, 'ChunkIndex'], nchunks: Union[np.ndarray,List[int]]) -> np.ndarray:
        nchunks = np.asarray(nchunks,dtype=int)
        # Catch the special case where NDIM = 1 and we have a singleton value.
        if (len(nchunks) == 1) and (isinstance(index, int)):
            index = [index]
        elif isinstance(index,int):
            raise ValueError(f"Failed to coerce chunk index ({index}):\n"
                             "The index was an integer, but NDIM > 1.")

        # CONVERT to np array to avoid any type issues.
        obj = np.asarray(index,dtype=int).view(cls)

        if len(obj) != len(nchunks):
            raise ValueError(f"Failed to coerce chunk index ({index}):\n"
                             f"The index had length {len(obj)}, but NDIM={len(nchunks)}.")

        # Validate that the chunk index is in the right range.
        _chunk_base = np.zeros_like(nchunks)

        if np.any(np.greater_equal(obj,nchunks) | np.less(obj,_chunk_base)):
            raise ValueError(f"Failed to coerce chunk index ({index}):\n"
                             f"Index was not on chunk range {_chunk_base} to {nchunks - 1}.")

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
        if type(self) is not ChunkIndex:
            return np.asarray(self)