"""
Base class grid managers for Pisces.

This module provides the core classes and logic for defining, storing, and manipulating
the structured grids that underlie Pisces models.

For details on the use of this module, consult :ref:`model_grid_management`.


"""
from pathlib import Path
from typing import Union, Optional, List, Iterable, Iterator, Callable, TYPE_CHECKING

import h5py
import numpy as np
import unyt
from numpy.typing import NDArray, ArrayLike
from tqdm.auto import tqdm

from pisces.geometry import GeometryHandler
from pisces.geometry.coordinate_systems import CoordinateSystem
from pisces.io import HDF5_File_Handle, HDF5ElementCache
from pisces.models.grids.structs import BoundingBox, DomainDimensions, ChunkIndex
from pisces.utilities.array_utils import make_grid_fields_broadcastable
from pisces.utilities.config import pisces_params
from pisces.utilities.logging import devlog

if TYPE_CHECKING:
    from pisces.profiles.base import Profile

AxesSpecifier = Iterable['AxisAlias']
AxesMask = NDArray[np.bool_]

class ModelGridManager:
    r"""
    Manager class controlling the coordinate grid in Pisces models.

    The :py:class:`ModelGridManager` manages the underlying coordinate grid and
    component :py:class:`ModelField`-s of all Pisces :py:class:`~pisces.models.base.Model` instances.

    A :py:class:`ModelGridManager` provides the essential infrastructure to work with
    multi-dimensional data (fields) on a structured grid, including:

    - The bounding box (:py:attr:`BBOX`) describing the physical extent of each axis.
    - The grid shape (:py:attr:`GRID_SHAPE`) specifying how many cells exist along each axis.
    - Chunk shapes (:py:attr:`CHUNK_SHAPE`) for memory-efficient partial reads/writes.
    - A coordinate system (:py:attr:`coordinate_system`) to define axis names and transformations.
    - Mechanisms for chunked operations, coordinate generation, and new field creation.

    The manager internally references an HDF5 file at :py:attr:`path`. If the file does not exist,
    a new one is created (given sufficient parameters such as ``bbox`` and ``grid_shape``); if it does
    exist, a manager attempts to load its metadata from that file.

    .. note::

       If you need to recreate a file from scratch (deleting an existing file), use ``overwrite=True``
       when instantiating. This permanently removes any prior data stored at that location.

    See Also
    --------
    ModelField : The individual data arrays stored in the manager.
    ModelFieldContainer : The container managing all fields in the HDF5 file.

    """
    # @@ CLASS ATTRIBUTES @@ #
    # These flags / defaults can be set in subclasses to
    # refine the standard behavior of subclasses.
    DEFAULT_LENGTH_UNIT: Union[str, unyt.Unit] = 'kpc'
    """ str or unyt.Unit: The default length unit for this class.
    
    The default length unit can be overwritten / bypassed by providing the ``length_unit``
    argument when initializing this class.
    """
    DEFAULT_SCALE: Union[str, List[str]] = 'linear'
    """ str or list of str: The default scale for this class.
    
    Unless provided during class generation, the default scale determines the :py:attr:`scale` of the
    grid manager.
    """
    ALLOWED_COORDINATE_SYSTEMS: Optional[List[str]] = None
    """ list of str: The names of the permitted coordinate systems for this manager class.
    
    If the class does not specify any coordinate systems here, then all coordinate systems are permitted.
    """

    def __init__(self,
                 path: Union[str, Path],
                 /,
                 coordinate_system: CoordinateSystem = None,
                 bbox: ArrayLike = None,
                 grid_shape: ArrayLike = None,
                 chunk_shape: ArrayLike = None,
                 *,
                 overwrite: bool = False,
                 length_unit: str = None,
                 scale: Union[List[str], str] = None):
        """
        Initialize a :py:class:`ModelGridManager` instance.

        Parameters
        ----------
        path : Union[str, Path]
            The path to the HDF5 file.

            If the path exists, then an attempt is made to load it as a manager; otherwise a new manager is initialized
            in a new file of the same name.

            .. tip::

                If you want to overwrite an existing instance, you need to use ``overwrite=True``.

        coordinate_system : :py:class:`~pisces.geometry.base.CoordinateSystem`, optional
            The coordinate system that defines the dimensionality and axes of the grid.

            This is required when creating a new file (i.e., if the HDF5 file does not
            already exist). If you open an existing file, the coordinate system is loaded
            automatically from that file.

        bbox : NDArray[np.float64], optional
            The bounding box that defines the physical extent of the grid in each axis.
            Required if you are creating a new file. This should be convertible into a
            :py:class:`~pisces.models.grids.structs.BoundingBox`, with shape ``(2, NDIM)``,
            where ``NDIM`` matches the number of axes in ``coordinate_system``. The first row
            contains the minimum coordinate values along each axis, and the second row
            contains the maximum coordinate values.

            .. note::

               You can provide this in a Python list form such as ``[[x0_min, x0_max],
               [x1_min, x1_max], ...]``.

        grid_shape : NDArray[np.int64], optional
            The shape of the grid, specifying the number of cells along each axis.
            This is required for creating a new file. It should be a 1D array-like of
            integers with length equal to the number of dimensions in
            ``coordinate_system``.
        chunk_shape : NDArray[np.int64], optional
            The shape of each chunk used for subdividing the grid, allowing chunk-based
            operations or partial in-memory loading.

            If not provided, it defaults to
            ``grid_shape``, meaning the entire grid is treated as a single chunk. If specified,
            each element must divide the corresponding element in ``grid_shape`` without
            remainder.

            .. important::

                The choice to perform operations in chunks (or not to) is made by the developer of
                the relevant model. Generally, if it isn't necessary to perform operations in chunks, it's avoided.
                As such, it's generally advisable to leave this argument unchanged unless you have a clear reason
                to set it.

        overwrite : bool, optional
            If ``True``, an existing HDF5 file at ``path`` is removed before creating a new one.
            Defaults to ``False``. Note that this cannot overwrite an open or locked file.

        length_unit : str, optional
            The physical length unit for interpreting grid coordinates, for example `"kpc"`
            or `"cm"`. Defaults to :py:attr:`DEFAULT_LENGTH_UNIT`.
        scale : Union[List[str], str], optional
            The scaling mode for each axis, determining whether cells are spaced linearly or
            logarithmically. Each entry can be `"linear"` or `"log"`. If a single string is given,
            it is applied to all axes. Defaults to :py:attr:`DEFAULT_SCALE`.

        Raises
        ------
        ValueError
            If required parameters are missing when creating a new file or if dimensions are invalid.
        """
        # SETUP path and manage the overwrite procedure.
        self._path: Path = Path(path)
        if self.path.exists() and overwrite:
            self.path.unlink()

        # MANAGE the setup of a new instance of one doesn't already exist.
        #   If the path exists, this is skipped, and we just load; otherwise, we
        #   attempt to generate a new instance using the skeleton builder.
        if not self.path.exists():
            # enforce required arguments.
            if any(arg is None for arg in [bbox, grid_shape, coordinate_system]):
                raise ValueError(
                    f"Cannot create new ModelGridManager at {path} because not all "
                    f"of `bbox`, `grid_shape`, and `coordinate_system` were provided."
                )

            # Build the skeleton
            self._handle = HDF5_File_Handle(self.path, mode='w')
            self.build_skeleton(
                self._handle,
                coordinate_system,
                bbox,
                grid_shape,
                chunk_shape,
                length_unit=length_unit,
                scale=scale
            )
            # Switch the handle so that we can read data as well.
            self._handle = self._handle.switch_mode('r+')
        else:
            # Open an existing file
            self._handle = HDF5_File_Handle(self.path, mode='r+')

        # LOAD and compute attributes.
        #   These methods load the relevant attributes from the HDF5 file structure
        #   and then compute derived attributes from them on the fly. They can be
        #   safely reimplemented in subclasses.
        self._load_attributes()
        self._compute_secondary_attributes()

        # Load fields
        self._load_fields()

    # @@ LOADERS @@ #
    # These methods are all featured in the __init__ call as
    # modular methods to allow easy subclassing and keep things more
    # readable.
    #
    # There do frequently need to be altered in subclasses. Developers
    # should read through each of them and determine if changes need to be made.
    def _load_attributes(self):
        """
        Load grid attributes from the HDF5 file.

        Notes
        -----
        This method loads bounding box, chunk shape, grid shape, scaling, and length unit from the file.
        """
        # Load the core attributes saved in the handle attributes.

        #: Docstring for bbox.
        self._BBOX: BoundingBox = BoundingBox(self.handle.attrs["BBOX"])
        self._CHUNK_SHAPE = DomainDimensions(self.handle.attrs["CHUNK_SHAPE"])
        self._GRID_SHAPE = DomainDimensions(self.handle.attrs["GRID_SHAPE"])
        self._scale: List[str] = self.handle.attrs["SCALE"]
        self._length_unit: unyt.Unit = unyt.Unit(self.handle.attrs["LUNIT"])

        # Load the coordinate system from the handle coordinates system group.
        self._coordinate_system: CoordinateSystem = CoordinateSystem.from_file(self.handle['CSYS'], fmt='hdf5')

        # ensure that the coordinate system is a valid coordinate system.
        if self.__class__.ALLOWED_COORDINATE_SYSTEMS is not None:
            if self._coordinate_system.__class__.__name__ not in self.__class__.ALLOWED_COORDINATE_SYSTEMS:
                raise ValueError(f'Attempted to load invalid GridManager with coordinate system class {self._coordinate_system.__class__.__name__}.\n'
                                 f'{self.__class__.__name__} only supports {self.__class__.ALLOWED_COORDINATE_SYSTEMS}.')

    def _load_fields(self):
        """
        Load the fields associated with the grid.

        Notes
        -----
        Fields are managed using the `ModelFieldContainer` for easy access and manipulation.
        """
        self._fields = ModelFieldContainer(self)

    def _compute_secondary_attributes(self):
        """
        Compute attributes derived from grid metadata.

        Notes
        -----
        - Computes scaled bounding box and log mask based on scaling.
        - Computes chunk size and cell size in scaled units.
        """
        # Manage scaling
        self._log_mask = np.array([ax == 'log' for ax in self.scale], dtype=bool)
        self._scaled_bbox = self.BBOX[...]
        self._scaled_bbox[:, self._log_mask] = np.log10(self._scaled_bbox[:, self._log_mask])

        # Compute chunk and cell sizes
        self._NCHUNKS = self.GRID_SHAPE // self.CHUNK_SHAPE
        self._CHUNK_SIZE = (self._scaled_bbox[1, :] - self._scaled_bbox[0, :]) / self._NCHUNKS
        self._CELL_SIZE = self._CHUNK_SIZE / self.CHUNK_SHAPE

    @classmethod
    def build_skeleton(cls,
                       handle: HDF5_File_Handle,
                       coordinate_system: CoordinateSystem,
                       bbox: NDArray[np.floating],
                       grid_shape: NDArray[np.int_],
                       chunk_shape: Optional[NDArray[np.int_]] = None,
                       length_unit: str = None,
                       scale: Union[List[str] | str] = None,
                       ) -> HDF5_File_Handle:
        """
        Construct a "skeleton" for the :py:class:`ModelGridManager` class.

        The skeleton is the base structure necessary to load an HDF5 file as this object.

        Parameters
        ----------
        handle: :py:class:`~pisces.io.HDF5_File_Handle`
            The HDF5 file handle at which to construct the skeleton.

        coordinate_system : :py:class:`~pisces.geometry.base.CoordinateSystem`, optional
            The coordinate system that defines the dimensionality and axes of the grid.

        bbox : NDArray[np.float64], optional
            The bounding box that defines the physical extent of the grid in each axis.
            This should be convertible into a :py:class:`~pisces.models.grids.structs.BoundingBox`,
            with shape ``(2, NDIM)``, where ``NDIM`` matches
            the number of axes in ``coordinate_system``. The first row
            contains the minimum coordinate values along each axis, and the second row
            contains the maximum coordinate values.

            .. note::

               You can provide this in a Python list form such as ``[[x0_min, x0_max],
               [x1_min, x1_max], ...]``.

        grid_shape : NDArray[np.int64], optional
            The shape of the grid, specifying the number of cells along each axis. It should be a 1D array-like of
            integers with length equal to the number of dimensions in
            ``coordinate_system``.
        chunk_shape : NDArray[np.int64], optional
            The shape of each chunk used for subdividing the grid, allowing chunk-based
            operations or partial in-memory loading.

            If not provided, it defaults to
            ``grid_shape``, meaning the entire grid is treated as a single chunk. If specified,
            each element must divide the corresponding element in ``grid_shape`` without
            remainder.

            .. important::

                The choice to perform operations in chunks (or not to) is made by the developer of
                the relevant model. Generally, if it isn't necessary to perform operations in chunks, it's avoided.
                As such, it's generally advisable to leave this argument unchanged unless you have a clear reason
                to set it.

        length_unit : str, optional
            The physical length unit for interpreting grid coordinates, for example `"kpc"`
            or `"cm"`. Defaults to :py:attr:`DEFAULT_LENGTH_UNIT`.
        scale : Union[List[str], str], optional
            The scaling mode for each axis, determining whether cells are spaced linearly or
            logarithmically. Each entry can be `"linear"` or `"log"`. If a single string is given,
            it is applied to all axes. Defaults to :py:attr:`DEFAULT_SCALE`.

        Raises
        ------
        ValueError
            If required parameters are missing when creating a new file or if dimensions are invalid.

        """
        # Perform validation tasks. Ensure that the bounding box, grid_shape, chunk_shape, etc.
        # are correctly formatted and meet all the necessary standards.
        #
        # Validate the coordinate system.
        if cls.ALLOWED_COORDINATE_SYSTEMS is not None:
            # The allowed coordinate systems need to be checked.
            if coordinate_system.__class__.__name__ not in cls.ALLOWED_COORDINATE_SYSTEMS:
                raise ValueError(f"Failed to build skeleton for {cls.__name__}:\n"
                                 f"Input coordinate system was a {coordinate_system.__class__.__name__} instance, but "
                                 f"{cls.__name__} only supports the following: {cls.ALLOWED_COORDINATE_SYSTEMS}.")

        coordinates_ndim = coordinate_system.NDIM
        bbox, grid_shape = BoundingBox(bbox), DomainDimensions(grid_shape)

        if chunk_shape is None:
            # Set the chunk shape to be the same as the grid shape.
            chunk_shape = grid_shape

        # ensure that the chunk shape is a valid DomainDimensions object.
        chunk_shape = DomainDimensions(chunk_shape)

        # check that the dimensions are uniform.
        if len({bbox.shape[-1], len(grid_shape), len(chunk_shape), coordinates_ndim}) != 1:
            raise ValueError(
                f"Detected inconsistent dimensions while building skeleton: dimensions for bbox, grid_shape,"
                f" chunk_shape, and coordinate system were"
                f" {[bbox.shape[-1], len(grid_shape), len(chunk_shape), coordinates_ndim]} respectively.")

        # Check that the grid shape can be divided by the chunk shape.
        # This ensures that we don't get any partial chunks.
        # Check that the chunks fit
        if np.any(grid_shape % chunk_shape != 0):
            raise ValueError(f"Grid shape {grid_shape} must be divisible by chunk shape {chunk_shape}."
                             f" Pisces does not support partial chunking.")

        # Coerce the scale to be a format that we like.
        if scale is None:
            # The scale needs to be set to the default. We check for instance type to ensure
            # that we do not provide access to a mutable attribute of the class.
            if isinstance(cls.DEFAULT_SCALE, list):
                scale = cls.DEFAULT_SCALE[:]
            else:
                scale = cls.DEFAULT_SCALE

        if isinstance(scale, str):
            # Alter the scale to be an iterable of the correct length.
            scale = [scale] * coordinate_system.NDIM
        else:
            if len(scale) != coordinate_system.NDIM:
                raise ValueError(f"Scale {scale} must be specified for all dimensions.")

        if any(k not in ['linear', 'log'] for k in scale):
            raise ValueError(f"Scale {scale} must be linear or log.")

        # Validate the length unit.
        if length_unit is None:
            length_unit = cls.DEFAULT_LENGTH_UNIT

        # Setup the attributes and construct the structure.
        # All validation tasks are complete, we can now proceed with the generation of
        # the structure.
        #
        # Add the coordinate system to the file at the CSYS group position.
        coordinate_system.to_file(handle.require_group('CSYS'), fmt='hdf5')

        # WRITE parameters to disk
        handle.attrs['LUNIT'] = str(length_unit)
        handle.attrs['SCALE'] = scale
        handle.attrs['CHUNK_SHAPE'] = chunk_shape
        handle.attrs['BBOX'] = bbox
        handle.attrs['GRID_SHAPE'] = grid_shape

        return handle

    # @@ DUNDER METHODS @@ #
    # These are basic dunder methods. They should not be
    # altered in subclasses to preserve base-level functionality
    # across subclass structures.
    def __str__(self):
        return f"<{self.__class__.__name__}: {self.path}>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self,field: str) -> 'ModelField':
        return self.FIELDS[field]

    def __setitem__(self,field: str, value: NDArray[np.floating]) -> None:
        self.FIELDS[field][...] = value

    def __len__(self):
        """Return the number of fields."""
        return len(self.FIELDS)

    def __contains__(self, field: str) -> bool:
        """Check if a field exists in the grid."""
        return field in self.FIELDS

    def __iter__(self):
        """Iterate over all fields in the grid."""
        return iter(self.FIELDS)

    # @@ COORDINATE MANAGEMENT @@ #
    # These methods are used for coordinate determinations.
    def get_coordinates(
            self,
            chunk_index: Optional[ChunkIndex] = None,
            axes: Optional[AxesSpecifier] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the cell-centered coordinates for the grid or a specific chunk.

        Parameters
        ----------
        chunk_index : :py:class:`~pisces.models.grids.structs.ChunkIndex`, optional
            The index of the chunk for which coordinates are computed. If `None`, compute for the entire grid.
        axes : Optional[AxesSpecifier], optional
            The axes for which coordinates are computed. If `None`, use all axes.

        Returns
        -------
        NDArray[np.float64]
            An array of cell-centered coordinates with shape ``(*GRID_SHAPE, len(axes))`` (see :py:attr:`GRID_SHAPE`).
            If a value is provided for ``chunk_index``, then the returned array will have shape ``(*CHUNK_SHAPE, len(axes))``
            (see :py:attr:`CHUNK_SHAPE`).

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        # Validation. Ensure that we have a set of axes. Construct the axes mask.
        if axes is None:
            axes = self.coordinate_system.AXES
        axes_mask = self.coordinate_system.build_axes_mask(axes)

        # Look up the relevant bounding box for either the chunk or the entire grid.
        # We want this in scaled axes so that we can later correct it for the log-ed axes.
        if chunk_index is not None:
            chunk_index = ChunkIndex(chunk_index, self.NCHUNKS[axes_mask])
            bbox = self.get_chunk_bbox(chunk_index,axes=axes,scale=True)
            shape = self.CHUNK_SHAPE[axes_mask]
        else:
            bbox = self.SCALED_BBOX[:,axes_mask]
            shape = self.GRID_SHAPE[axes_mask]

        cell_size = self.CELL_SIZE[axes_mask]

        # Construct the slices. These should run (in the scaled space) from the bottom
        # of the bbox to the top in increments of the cell_size (scaled).
        # We only build slices for the coordinates we actually want.
        slices = [
            slice(bbox[0, i] + 0.5 * cell_size[i],
                  bbox[1, i] - 0.5 * cell_size[i],
                  shape[i] * 1j)
            for i in np.arange(len(axes))
        ]

        # Create the coordinate grid. This requires moving the axis to the
        # correct position and then rescaling for the log components.
        coordinates = np.moveaxis(np.mgrid[*slices], 0, -1)

        coordinates[..., self.is_log_mask[axes_mask]] = 10**coordinates[..., self.is_log_mask[axes_mask]]
        return coordinates

    # @@ CHUNK UTILITIES @@ #
    # These methods are utilities for performing operations in
    # chunks.
    def get_chunk_bbox(self,
                       chunk_index: np.ndarray,
                       axes: Optional[List[str]] = None,
                       scale: bool = False) -> BoundingBox:
        """
        Compute the bounding box for a specific.

        Parameters
        ----------
        chunk_index: np.ndarray
            The index of the chunk. Should be an iterable of length ``NDIM`` with the index of
            the chunk along each axis.
        axes: Optional[List[str]], optional
            The axes along which the ``chunk_index`` is specified. By default, we assume a complete chunk index.
        scale: bool, optional
            If ``True``, will return the scaled bounding box section (see :py:attr:`SCALED_BBOX`). Otherwise, will
            return the standard bbox.
        Returns
        -------
        BoundingBox
            The bounding box of the chunk in scaled coordinates.

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        # Validate the axes.
        if axes is None:
            axes = self.coordinate_system.AXES
        axes_mask = self.coordinate_system.build_axes_mask(axes)
        # Ensure the chunk index is valid.
        chunk_index = ChunkIndex(chunk_index, self.NCHUNKS[axes_mask])

        # Pull out the positions. This is taken from the scaled bounding box because
        # the chunks have linear sizes in the scaled domain.
        left = self._scaled_bbox[0, axes_mask] + chunk_index * self.CHUNK_SIZE[axes_mask]
        right = left + self.CHUNK_SIZE[axes_mask]

        # Build the base of the bbox.
        _bbox = np.stack((left, right), axis=-1)

        # Account for scaling
        if scale:
            return BoundingBox(_bbox)
        else:
            _bbox[:,self.is_log_mask[axes_mask]] = 10**_bbox[:,self.is_log_mask[axes_mask]]
            return BoundingBox(_bbox)

    def get_chunk_mask(self, chunk_index: ChunkIndex, axes: Optional[AxesSpecifier] = None) -> List[slice]:
        """
        Construct the mask of the full grid corresponding to a specific ``chunk_index``.

        Given a set of ``axes``, the total grid has some shape ``(N_0,...,N_k)``. A chunk occupies some subset of
        of that grid space. This function produces a list of slices to select only the relevant section corresponding
        to the desired chunk.

        Parameters
        ----------
        chunk_index : ChunkIndex
            The index of the chunk.
        axes : AxesSpecifier
            The axes for which the slice indices are computed.

        Returns
        -------
        list of slice
            The set of slices corresponding to the ``chunk_index``. Each element of the returned list corresponds
            to the corresponding slice in the grid along the specific axis specified in ``axes``.

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        if axes is None:
            axes = self.coordinate_system.AXES
        axes_mask = self.coordinate_system.build_axes_mask(axes)
        chunk_index = ChunkIndex(chunk_index,self.NCHUNKS[axes_mask])

        start = chunk_index * self.CHUNK_SHAPE[axes_mask]
        end = (chunk_index + 1) * self.CHUNK_SHAPE[axes_mask]

        return [slice(s, e) for s, e in zip(start, end)]

    def iterate_over_chunks(self, axes: Optional[List[str]] = None) -> Iterator[ChunkIndex]:
        """
        Iterate over all chunks along the specified axes.

        Parameters
        ----------
        axes : Optional[List[str]], optional
            The axes for which chunks are iterated. If `None`, use all axes.

        Yields
        ------
        NDArray[np.int_]
            The chunk index for each iteration.
        """
        # Manipulate the axes and generate the axes mask.
        if axes is None:
            axes = self.coordinate_system.AXES
        axes_mask = self.coordinate_system.build_axes_mask(axes)

        # Construct the index arrays. np.indices generates array of shape (NDIM,*self.NCHUNKS) which can then
        # be collapsed to (NDIM, TOTAL_CHUNKS) to contain all of the chunk indices.
        index_array = np.indices(self.NCHUNKS[axes_mask],dtype=int) # (NDIM, *self.NCHUNKS)
        index_array = index_array.reshape(len(self.NCHUNKS[axes_mask]),-1) # (NDIM, TOTAL_CHUNKS)
        index_array = index_array.T # (TOTAL_CHUNKS,NDIM)

        for chunk_index in index_array:
            yield chunk_index

    # @@ UTILITY FUNCTIONS @@ #
    # These methods provide backend utilities for various processes
    # underlying chunk integration and operation management.
    def make_fields_consistent(self, arrays: List[np.ndarray], axes: List[Union[List[str]]]) -> List[np.ndarray]:
        """
        Ensures that multiple arrays are broadcastable based their axes.

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
        """
        return make_grid_fields_broadcastable(arrays,axes,coordinate_system=self.coordinate_system)

    # @@ FEATURES @@ #
    def add_field_from_function(
            self,
            function: Callable,
            field_name: str,
            /,
            axes: Optional[List[str]] = None,
            *,
            chunking: bool = False,
            units: Optional[str] = '',
            dtype: str = "f8",
            overwrite: bool = False,
            **kwargs,
    ):
        """
        Create a :py:class:`ModelField` in this :py:class:`ModelGridManager` by evaluating the provided function.

        This method takes a function (``function``) and evaluates it at the relevant grid points to generate
        a new field with name ``field_name``.

        Parameters
        ----------
        function : Callable
            A function which takes (as input) ``N`` arguments ``(x_1,...,x_N)`` corresponding to the coordinate
            values of the ``N`` axes specified by the ``axes`` argument. If ``axes`` is not specified, then ``N=NDIM``, where
            ``NDIM`` is the number of dimensions in the coordinate system.
        field_name : str
            The name to give to the newly generated field.

            .. note::

                The ``field_name`` will be the location in the HDF5 file as well (``FIELDS/field_name``).

        axes : Optional[List[str]], optional
            The coordinate axes along which the function is to be evaluated. If ``axes`` is not provided, then
            it is assumed that the function operates on all the coordinates of the coordinate system.

            .. hint::

                Ensure that ``axes`` is self-consistent with the call signature of the ``function`` parameter.

        chunking : bool, optional
            If `True`, evaluate the function in chunks. Default is `False`.

            .. tip::

                This is generally not necessary unless you cannot load the entire base grid into memory at once. This
                is particularly common if the function is operating in 3 or more dimensions, in which case even moderately
                resolved grids may take up significant memory.

        units : Optional[str], optional
            The units to give to the field. If ``units`` is not provided, then it is assumed that the field is
            dimensionless.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.

        Raises
        ------
        ValueError
            If the function, axes, or other parameters are invalid.
        """
        # Create the base field (empty) in the FIELDS collection.
        # This ensures that we can proceed to populate the field within this method.
        self.FIELDS.add_field(
            field_name,
            axes=axes,
            units=units,
            dtype=dtype,
            overwrite=overwrite,
        )

        # Create a reference to the _field object so that we don't have
        # to load the entire object into memory at once.
        _field = self.FIELDS[field_name]

        if not chunking:
            # We can proceed without chunking and just set the entire field at once.
            coordinates = self.get_coordinates(axes=axes)
            _field[...] = function(*np.moveaxis(coordinates, -1, 0))
        else:
            # We are going to use chunks. We now need to iterate through each chunk.
            _progress_bar = tqdm(desc=f'(Chunked) Building field {field_name}',
                                 total=int(np.prod(self.NCHUNKS[axes])),
                                 leave = False,
                                 disable= pisces_params['system.preferences.disable_progress_bars'])
            for _ci in self.iterate_over_chunks(axes):
                _chunk_mask, _chunk_coordinates = self.get_chunk_mask(_ci,axes=axes), self.get_coordinates(chunk_index=_ci,axes=axes)
                _field[*_chunk_mask] = function(*np.moveaxis(_chunk_coordinates, -1, 0))
                _progress_bar.update(1)
            _progress_bar.close()

    def add_field_from_profile(self,
                               profile: 'Profile',
                               field_name: str,
                               /,
                               *,
                               chunking: bool = False,
                               units: Optional[str] = None,
                               dtype: str = "f8",
                               overwrite: bool = False,
                               **kwargs):
        """
        Create a :py:class:`ModelField` in this :py:class:`ModelGridManager` by evaluating the provided profile.

        This method effectively wraps the :py:meth:`ModelGridManager.add_field_from_function` but utilizes the axes
        information from the profile to reduce the number of necessary inputs.

        Parameters
        ----------
        profile : :py:class:`~pisces.profiles.base.Profile`
            Any valid :py:class:`~pisces.profiles.base.Profile` instance.

            .. hint::

                To be a valid :py:class:`~pisces.profiles.base.Profile` instance, the profile must have the same
                axes as :py:attr:`coordinate_system` or have axes which are a subset of them.

        field_name : str
            The name to give to the newly generated field.

            .. note::

                The ``field_name`` will be the location in the HDF5 file as well (``FIELDS/field_name``).

        chunking : bool, optional
            If `True`, evaluate the function in chunks. Default is `False`.

            .. tip::

                This is generally not necessary unless you cannot load the entire base grid into memory at once. This
                is particularly common if the function is operating in 3 or more dimensions, in which case even moderately
                resolved grids may take up significant memory.

        units : Optional[str], optional
            The units to give to the field. If ``units`` is not provided, then it is assumed that the field is
            dimensionless.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.

        Raises
        ------
        ValueError
            If the function, axes, or other parameters are invalid.
        """
        # Validate that the profile's axes align with the grid's coordinate system
        if not all(ax in self.coordinate_system.AXES for ax in profile.AXES):
            raise ValueError(
                f"Profile axes {profile.AXES} are incompatible with the grid's coordinate system axes: {self.coordinate_system.AXES}."
            )

        # Determine field units
        if units is None:
            units = profile.units

        # Validate unit compatibility
        if units != profile.units:
            try:
                conv_factor = unyt.Unit(units).get_conversion_factor(profile.units)[0]
            except Exception as e:
                raise ValueError(f"Unit conversion error: {e}.") from e
        else:
            conv_factor = 1

        # Delegate to `add_field_from_function`
        self.add_field_from_function(
            lambda *coords: profile(*coords) * conv_factor,
            field_name,
            axes=profile.AXES,
            chunking=chunking,
            units=units,
            dtype=dtype,
            overwrite=overwrite,
            **kwargs,
        )

    def get_grid_summary(self):
        """
        Generate a summary of the grid structure used in the model.

        This summary includes:
        - Axis Name
        - Minimum and Maximum values on each axis
        - Number of grid cells and chunks
        - Grid cell size and chunk size
        - Scaling type (linear or logarithmic)

        Returns
        -------
        Union[List[List[Any]], str]
            - If the ``tabulate`` library is installed, the summary is returned as a formatted table string.
            - If ``tabulate`` is not available, the summary is returned as a nested list of grid information.

        Notes
        -----
        - The ``BBOX`` attribute provides the bounding box for the grid.
        - The ``GRID_SHAPE`` specifies the total number of grid cells along each axis.
        - The ``CHUNK_SHAPE`` determines how the grid is split into chunks.
        - The ``CELL_SIZE`` and ``CHUNK_SIZE`` attributes reflect the spatial size of each grid cell and chunk.
        - Scaling type is either ``log`` (logarithmic) or ``lin`` (linear). If the scale is ``log``, values are adjusted to base-10 representation.

        See Also
        --------
        pisces.models.grids.base.ModelGridManager : Manages grid structure and provides access to grid metadata.

        Examples
        --------
        The output will look something like this when `tabulate` is installed:

        .. code-block:: python

            +------+---------+---------+-----+-----------+----------------+----------------+-------+
            | Axis | Min.    | Max.    |  N  | N Chunks  | Cell Size      | Chunk Size     | Scale |
            +------+---------+---------+-----+-----------+----------------+----------------+-------+
            |   r  | 1.00e+00| 1.00e+03| 100 |     10    | 1.00e+00 - lin | 1.00e+01 - lin | lin   |
            | theta| 0.00e+00| 3.14e+00|  50 |      5    | 6.28e-02 - lin | 3.14e-01 - lin | lin   |
            |  phi | 0.00e+00| 6.28e+00|  72 |      9    | 8.73e-02 - lin | 5.24e-01 - lin | lin   |
            +------+---------+---------+-----+-----------+----------------+----------------+-------+


        If `tabulate` is not installed, the method will return raw data:

        .. code-block:: python

            [
                ["r", "1.00e+00", "1.00e+03", 100, 10, "1.00e+00 - lin", "1.00e+01 - lin", "lin"],
                ["theta", "0.00e+00", "3.14e+00", 50, 5, "6.28e-02 - lin", "3.14e-01 - lin", "lin"],
                ["phi", "0.00e+00", "6.28e+00", 72, 9, "8.73e-02 - lin", "5.24e-01 - lin", "lin"]
            ]

        """
        # Import the tabulate method that we need to successfully run this.
        try:
            from tabulate import tabulate
            _use_tabulate = True
        except ImportError:
            _use_tabulate = False
            tabulate = None  #! TRICK the IDE

        axes_info = []
        for axi, ax in enumerate(self.coordinate_system.AXES):
            amin, amax = self.BBOX[0, axi], self.BBOX[1, axi]
            Ngrid, Nchunk = self.GRID_SHAPE[axi], (self.GRID_SHAPE // self.CHUNK_SHAPE)[axi]
            Sgrid, Schunk = self.CELL_SIZE[axi], self.CHUNK_SIZE[axi]
            scale = self.scale[axi]

            if scale == 'log':
                amin, amax = np.format_float_scientific(float(10**amin), precision=3, unique=True), \
                             np.format_float_scientific(float(10**amax), precision=3, unique=True)
                Sgrid = f"{np.format_float_scientific(float(Sgrid), precision=2)} - log"
                Schunk = f"{np.format_float_scientific(float(Schunk), precision=2)} - log"
            else:
                amin, amax = np.format_float_scientific(float(amin), precision=2), \
                             np.format_float_scientific(float(amax), precision=2)
                Sgrid = f"{np.format_float_scientific(float(Sgrid), precision=2)} - lin"
                Schunk = f"{np.format_float_scientific(float(Schunk), precision=2)} - lin"

            axes_info.append([
                ax, amin, amax, Ngrid, Nchunk, Sgrid, Schunk, scale,
            ])

        if not _use_tabulate:
            return axes_info
        else:
            return tabulate(axes_info, headers=["Axis", "Min.", "Max.", "N", "N Chunks",
                                                "Cell Size", "Chunk Size", "Scale"], tablefmt="grid")

    # @@ PROPERTIES @@ #
    # For the most part, these all point to private
    # attributes elsewhere in the class structure. They shouldn't
    # need to be modified when generating a subclass.
    @property
    def FIELDS(self) -> 'ModelFieldContainer':
        """
        The physical fields in this manager file.

        The :py:attr:`FIELDS` attribute exposes the :py:class:`ModelFieldContainer` component of the :py:class:`ModelGridManager`
        allowing easy access to the physical fields in this manager. Each field is effectively a numpy array containing the
        data parameterizing one of the profiles in the model on a particular set of axes.

        Returns
        -------
        :py:class:`ModelFieldContainer`
            The container for managing the fields in the grid.
        """
        if self._fields is None:
            raise AttributeError("Fields container has not been initialized. Ensure the grid manager is correctly initialized.")
        return self._fields

    @property
    def BBOX(self) -> BoundingBox:
        """
        The bounding box describing the limits of the physical grid. The ``BBOX`` for the manager is a ``(2,NDIM)``
        array containing the minimum and maximum values of the grid along each of the axes of the coordinate space.

        For specific :py:class:`ModelField` instances within the manager, only a subset of these grid axes are necessarily
        present; however, they will always fill the ``BBOX`` in their relevant dimensions.

        .. tip::

            Intuitively, the :py:attr:`BBOX` determines the entire domain, then specific fields may simply be slices
            through that domain.

        Returns
        -------
        :py:class:`~pisces.models.grids.structs.BoundingBox`
        """
        return self._BBOX

    @property
    def GRID_SHAPE(self) -> DomainDimensions:
        """The shape of the full domain grid.

        The full grid shape determines how large field arrays will be in this :py:class:`ModelGridManager` instance.
        Each element of the grid shape corresponds to the number of cells in the grid along the axis matching that index.

        Returns
        -------
        :py:class:`~pisces.models.grids.structs.DomainDimensions`
        """
        return self._GRID_SHAPE

    @property
    def CHUNK_SHAPE(self) -> DomainDimensions:
        """
        The chunk shape of the grid.

        The chunk shape specifies the number of cells in each dimension which are present in any chunk of the domain. This
        parameter is a critical component of chunked-operations, which perform an operation on individual chunks to preserve
        memory.

        .. hint::

            The :py:attr:`CHUNK_SHAPE` attribute is set when the manager is created. If it is small, then chunked operations
            will take longer, but will have a lower maximum memory usage. For large chunks, the opposite is true.

            By default, we have a single chunk and everything is done in one pass (assuming your system has sufficient memory).

        Returns
        -------
        :py:class:`~pisces.models.grids.structs.DomainDimensions`

        """
        return self._CHUNK_SHAPE

    @property
    def scale(self) -> List[str]:
        """ The scaling between cells along each grid axis.

        Each element in :py:attr:`scale` is either ``"linear"`` or ``"log"``, indicating the spacing between
        adjacent cells along the corresponding grid axis.

        Returns
        -------
        list of str
        """
        return self._scale

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """The coordinate system associated with this manager.

        The coordinate system determines the number of dimensions for the grid, the different available axes, and various
        other parameters for the manager. It also provides capabilities for rapid transformations of the grid domain.

        Returns
        -------
        :py:class:`~pisces.models.grids.CoordinateSystem`
        """
        return self._coordinate_system

    @property
    def length_unit(self) -> unyt.Unit:
        """ The unit of length corresponding to the grid domain.

        :py:class:`ModelGridManagers` store the underlying grid as unitless values; these units allow for
        the grid domain, cell spacing, etc. to be connect to physical quantities correctly.

        Returns
        -------
        :py:class:`unyt.Unit`
        """
        return self._length_unit

    @property
    def path(self) -> Path:
        """
        The path associated with this manager.

        This is the file path to the underlying HDF5 file.

        Returns
        -------
        Path
        """
        return self._path

    @property
    def handle(self) -> HDF5_File_Handle:
        """
        The HDF5 file handle associated with this manager.

        Returns
        -------
        :py:class:`~pisces.io.hdf5.HDF5_File_Handle`
        """
        return self._handle

    @property
    def NCHUNKS(self) -> DomainDimensions:
        """
        The number of chunks along each axis of the grid.

        Each entry corresponds to how many chunks are laid out along
        the associated axis, given the total :py:attr:`GRID_SHAPE` and :py:attr:`CHUNK_SHAPE`.

        .. hint::

            If there :math:`N^i` cells along the :math:`i`-th dimension and chunks are :math:`n^i` in size, then
            the number of chunks is just

            .. math::

                N_{\rm chunks}^i = \frac{N^i}{n^i}.

        Returns
        -------
        DomainDimensions
            A :py:class:`~pisces.models.grids.structs.DomainDimensions` object
            specifying how many chunks exist for each axis.
        """
        return self._NCHUNKS

    @property
    def CHUNK_SIZE(self) -> NDArray[np.float64]:
        """
        The size of each chunk along every axis in scaled coordinates.

        This array has one entry per axis (like :py:attr:`GRID_SHAPE`), specifying
        how large each chunk is in the same scaling (log or linear) that
        the manager uses internally.

        Returns
        -------
        numpy.ndarray
            A 1D array of length ``NDIM`` giving the chunk size for each axis,
            in scaled coordinates.
        """
        return self._CHUNK_SIZE

    @property
    def CELL_SIZE(self) -> NDArray[np.float64]:
        """
        The size of each cell in scaled coordinates.

        Each cell’s size is determined by dividing :py:attr:`CHUNK_SIZE` by :py:attr:`CHUNK_SHAPE`.
        As such, this array has the same dimensionality as :py:attr:`CHUNK_SIZE`, giving
        the per-axis size for a single cell.

        Returns
        -------
        numpy.ndarray
            A 1D array of length ``NDIM`` specifying each cell’s extent in scaled units.
        """
        return self._CELL_SIZE

    @property
    def SCALED_BBOX(self) -> NDArray[np.float64]:
        """
        The bounding box in scaled coordinates.

        Whereas :py:attr:`BBOX` corresponds to the physical boundaries of the domain, :py:attr:`SCALED_BBOX`
        transforms any logarithmic axes (see :py:attr:`scale`) to their base-10 value. In the :py:attr:`SCALED_BBOX`,
        grid spacing and positions are all uniform, even if they are logarithmically scaled in the physical space.

        .. hint::

            This is most useful for developers; the idea is that you can streamline processes involving obtaining
            grid coordinates by "standardizing" coordinates and then just inverting the transformation at the end.

        Returns
        -------
        numpy.ndarray
            A ``(2, NDIM)`` array with the minimum and maximum scaled values
            of each axis in the grid.
        """
        return self._scaled_bbox

    @property
    def is_log_mask(self) -> NDArray[np.bool_]:
        """
        Indicates which axes are scaled logarithmically.

        A boolean array of length ``NDIM``, where each element is ``True``
        if the corresponding axis in :py:attr:`scale` is set to ``'log'``,
        and `False` otherwise.

        Returns
        -------
        numpy.ndarray
            A 1D boolean array with ``True`` entries for log-scaled axes,
            and `False` for linear-scaled axes.
        """
        return self._log_mask


class ModelField(unyt.unyt_array):
    """
    A field representing a physical profile in a :py:class:`ModelGridManager`.

    The :py:class:`ModelGridManager` acts as a collection of fields (:py:class:`ModelField`) along with a set of metadata
    including the relevant chunk, cell, and grid sizes; the coordinate system, etc. Each field in the manager represents
    a distinct physical profile over some set of dimensions.

    Notes
    -----
    The :py:class:`ModelField` class derives from the :py:class:`unyt.unyt_array` class with only a minor tweak:

        Sections / slices of fields are dynamically loaded from an underlying HDF5 dataset.

    This means that, even for very large arrays, data can be read chunk-by-chunk or simply sliced into without having
    to load the entire array.

    .. note::

        Generally speaking, this is done for you / in the backend of model generation routines. If you're developing a
        new model, this may be relevant to how you design your model. Most importantly, if your arrays are going to be
        large, take advantage of this architecture by operating in chunks instead of loading the entire array into
        memory at once.

    """

    def __new__(
            cls,
            manager: ModelGridManager,
            name: str,
            /,
            axes: Optional[List[str]] = None,
            data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
            *,
            overwrite: bool = False,
            dtype: str = "f8",
            units: str = "",
    ):
        # Build or retrieve the underlying HDF5 dataset.
        #    This ensures shape/units are validated and sets the dataset attributes.
        #    If the manager already has the dataset in question, it just gets returned here
        #    otherwise, we generate a new skeleton.
        dataset = cls.build_skeleton(
            manager,
            name,
            axes,
            data,
            overwrite=overwrite,
            dtype=dtype,
            units=units,
        )

        # Extract final units and axis ordering from the dataset's attributes.
        #    These might differ from user inputs if the skeleton had to coerce them.
        _units = dataset.attrs['units']
        _axes = dataset.attrs["axes"]

        # 3) Create the unyt array instance with placeholder data to avoid
        #    loading a large array into memory.
        #    We rely on on-disk slicing for actual data reads.
        obj = super().__new__(cls, [], units=units)

        # 4) Attach references to the newly constructed object.
        #    - `buffer` points to the on-disk dataset.
        #    - `_axes` records which axes this field depends on.
        #    - `_manager` points to the grid manager.
        #    - `_geometry_handler` is a dynamic loading reference for the geometry handler.
        obj._name = name
        obj.units = unyt.Unit(_units)
        obj.dtype = dataset.dtype
        obj.buffer = dataset
        obj._manager = manager
        obj._axes = _axes
        obj._geometry_handler = None

        return obj

    # noinspection PyUnusedLocal
    def __init__(
            self,
            manager: ModelGridManager,
            name: str,
            /,
            axes: Optional[List[str]] = None,
            data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
            *,
            overwrite: bool = False,
            dtype: str = "f8",
            units: str = "",
    ):
        """
        Construct a new :py:class:`ModelField` instance, backed by an HDF5 dataset.

        Parameters
        ----------
        manager : :py:class:`ModelGridManager`
            The grid manager that owns/coordinates this field structure.

            .. tip::

                The ``manager`` provides the coordinate system, the HDF5 access, and other
                "global" properties for the field.

        name : str
            The name of this field. This is (or will become) the name of the dataset in the
            underlying HDF5 group.
        axes : list of str, optional
            The axes (from the underlying :py:class:`~pisces.geometry.base.CoordinateSystem`) that are
            present in this field.

            The axes are used to determine the shape of the dataset based on the ``manager``'s
            grid shape, bounding box, and other attributes.

            .. note::

                ``axes`` are **always** reordered in the method to match the ordering native to the
                coordinate system for consistency.

        data : unyt.unyt_array or numpy.ndarray, optional
            Initial data to store in the newly generated dataset. This should be a ``(...,)`` array matching
            the expected shape of the grid determined from the ``manager``. If no data is provided, then the
            array is filled with nulls.

            .. tip::

                The ``manager`` has the :py:attr:`ModelGridManager.GRID_SHAPE`, which specifies the expected
                shape for each axis. The ``data`` should then match the shape along each of the axes that
                are specified in ``axes``.


        overwrite : bool, optional
            If True, an existing field/dataset with this name is deleted and replaced. Defaults to False.
        dtype : str, optional
            The NumPy dtype to use for storing the data. Defaults to 'f8'.
        units : str or unyt.Unit, optional
            Physical units of the field. If ``data`` is a unyt array with a
            different unit, an attempt is made to convert. Defaults to '' (unitless).

        Notes
        -----
        - This method calls :meth:`ModelField.build_skeleton` to either create or retrieve an
          existing dataset from the underlying HDF5 file.
        - The returned object is effectively a ``unyt.unyt_array`` but references
          an on-disk dataset for its storage. Slices can be read/written without
          loading the entire dataset into memory.

        Raises
        ------
        ValueError
            If data shape or units are incompatible with the manager's setup.
        """
        pass

    @classmethod
    def build_skeleton(
        cls,
        manager: ModelGridManager,
        name: str,
        /,
        axes: Optional[List[str]] = None,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        *,
        overwrite: bool = False,
        dtype: str = "f8",
        units: str = None,
    ) -> h5py.Dataset:
        """
        Construct a new :py:class:`ModelField` structure (an HDF5 dataset).

        If ``name`` is already in the ``FIELDS`` group of the ``manager``, then the
        existing dataset structure is returned. If not, then the other arguments are
        used to generate a new field structure in the correct position.

        .. tip::

            If you want to overwrite an existing field, you should use ``overwrite=True``.

        Parameters
        ----------
        manager : :py:class:`ModelGridManager`
            The grid manager that owns/coordinates this field structure.

            .. tip::

                The ``manager`` provides the coordinate system, the HDF5 access, and other
                "global" properties for the field.

        name : str
            The name of this field. This is (or will become) the name of the dataset in the
            underlying HDF5 group.
        axes : list of str, optional
            The axes (from the underlying :py:class:`~pisces.geometry.base.CoordinateSystem`) that are
            present in this field.

            The axes are used to determine the shape of the dataset based on the ``manager``'s
            grid shape, bounding box, and other attributes.

            .. note::

                ``axes`` are **always** reordered in the method to match the ordering native to the
                coordinate system for consistency.

        data : unyt.unyt_array or numpy.ndarray, optional
            Initial data to store in the newly generated dataset. This should be a ``(...,)`` array matching
            the expected shape of the grid determined from the ``manager``. If no data is provided, then the
            array is filled with nulls.

            .. tip::

                The ``manager`` has the :py:attr:`ModelGridManager.GRID_SHAPE`, which specifies the expected
                shape for each axis. The ``data`` should then match the shape along each of the axes that
                are specified in ``axes``.


        overwrite : bool, optional
            If True, an existing field/dataset with this name is deleted and replaced. Defaults to False.
        dtype : str, optional
            The NumPy dtype to use for storing the data. Defaults to 'f8'.
        units : str or unyt.Unit, optional
            Physical units of the field. If ``data`` is a unyt array with a
            different unit, an attempt is made to convert. Defaults to '' (unitless).

        Returns
        -------
        h5py.Dataset
            The dataset reference corresponding to the newly generated skeleton.

        Raises
        ------
        ValueError
            If data shape or units are incompatible with the manager's setup.
        """
        # Manage the HDF5 location / existence component of the procedure. If an
        # existing element is found a reference is checked against overwrite.
        handle = manager.handle.require_group("FIELDS") # This is the field storage location for all managers.
        if overwrite and name in handle:
            devlog.debug("ModelField - build_skeleton: removing %s from %s.",name,handle)
            del handle[name]

        # Look for (and return) an existing dataset of possible.
        if name in handle:
            return handle[name]

        # @@ AXES MANAGEMENT @@ #
        # Discover the coordinate system and check axes and data shape to ensure compatibility.
        # Ensure that axes are ordered correctly before moving forward with the data onboarding.
        _coordinate_system = manager.coordinate_system

        if axes is None:
            # copy the axes from the coordinate system.
            axes = _coordinate_system.AXES[:]

        if any(ax not in _coordinate_system.AXES for ax in axes):
            # We have unrecognized axes.
            raise ValueError(f"The following axes are not recognized for the {_coordinate_system.__class__.__name__}"
                             f" coordinate system: {[ax for ax in axes if ax not in _coordinate_system.AXES]}")

        # ensure axes are in order
        axes = _coordinate_system.ensure_axis_order(axes)

        # @@ SHAPE COORDINATION @@ #
        # We need to determine the correct shape. If data was actually provided,
        # we need to check that it has the right shape to proceed.
        axes_indices = np.array([_coordinate_system.ensure_axis_numeric(ax) for ax in axes],dtype=int) # Convert to indices.
        shape = manager.GRID_SHAPE[axes_indices]

        # @@ UNITS and DATA MANAGEMENT @@ #
        # This ensures that the units are handled correctly and that
        # we manage the data correctly as well.
        if data is not None:
            # Check the data for the correct shape.
            if not np.array_equal(data.shape,shape):
                raise ValueError(f"Failed to build ModelField skeleton:\n"
                                 f"Expected shape {shape} (axes={axes}) but received shape {data.shape}.")

            # If there is data provided to us, we need to also validate
            # that the data is what we expect it to be in terms of units and type.
            if isinstance(data, unyt.unyt_array):
                data_units = data.units
                if units is not None:
                    try:
                        data = data.to_value(units)
                    except Exception as e:
                        raise ValueError(
                            f"Inconsistent units: provided data has units '{data_units}', "
                            f"but specified units are '{units}'."
                        ) from e
                else:
                    units = data_units
                    data = data.to_value(data_units)
            else:
                if units is None:
                    units = ""  # --> set to default if not specified.

            # Enforce the datatype constraints.
            data = np.asarray(data, dtype=dtype)

        else:
            if units is None:
                units = ""  # --> set to default if not specified.

        # @@ DATASET CREATION @@ #
        # Create the relevant dataset and set the correct attributes.
        dataset = handle.create_dataset(name, shape=shape, data=data, dtype=dtype)

        # Set attributes
        dataset.attrs["units"] = str(units)
        dataset.attrs["axes"] = list(axes)

        return dataset

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override numpy's ufunc behavior to ensure operations return a `unyt_array`.

        Parameters
        ----------
        ufunc : numpy.ufunc
            The universal function to apply.
        method : str
            The ufunc method to use (e.g., "__call__").
        *inputs : tuple
            Input arrays for the ufunc.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        unyt_array
            The result of the operation with appropriate units.

        Notes
        -----
        Ensures unit consistency when performing numpy operations on `Field` objects.
        """
        # CAST ANY ModelField TO unyt_array FOR THE OPERATION
        # (THIS AVOIDS LOADING ENTIRE ARRAYS; STILL SLICES ON DEMAND)
        cast_inputs = tuple(
            (x.view(unyt.unyt_array) if isinstance(x, ModelField) else x)
            for x in inputs
        )
        result = getattr(ufunc, method)(*cast_inputs, **kwargs)

        # ATTACH THIS FIELD'S UNITS IF THE OPERATION YIELDS A unyt_array
        if isinstance(result, unyt.unyt_array):
            # This is simplistic: e.g. for addition of different fields,
            # unyt already handles unit consistency in the operation itself.
            # But we can ensure the final unit is the same as `self.units` if
            # the operation is dimensionally consistent.
            result.units = self.units

        return result


    def __setitem__(self, key: Union[slice, int], value: Union[unyt.unyt_array, np.ndarray]):
        """
        Set a slice of the field's data.

        Parameters
        ----------
        key : slice or int
            The indices to set.
        value : unyt.unyt_array or numpy.ndarray
            The value(s) to set.

        Raises
        ------
        ValueError
            If the units of the value are incompatible with the field's units.

        Notes
        -----
        Unit consistency is enforced when setting data. This ensures that
        the field's dataset always maintains valid physical units.
        """
        # Coerce units to ensure that we have
        # an unyt array to pass into the array.
        # Ensure that the units are self consistent.
        if isinstance(value, np.ndarray):
            value = unyt.unyt_array(value, self.units)
        elif not isinstance(value, unyt.unyt_array):
            raise ValueError(f"`value` must be a unyt_array or numpy.ndarray, not {type(value)}.")

        try:
            value = value.to_value(self.units)
        except Exception as e:
            raise ValueError(f"Failed unit conversion: {e}.") from e

        # Write the data to the HDF5 dataset
        self.buffer[key] = value

    def __getitem__(self, key: Union[slice, int]) -> unyt.unyt_array:
        """
        Retrieve a slice of data as a unyt_array with the field's units.
        """
        # GRAB THE RELEVANT PORTION DIRECTLY FROM DISK
        arr = self.buffer[key]  # => numpy ndarray
        return unyt.unyt_array(arr, units=self.units)

    @property
    def coordinate_system(self):
        """Shortcut to this field's manager's coordinate system."""
        return self._manager.coordinate_system

    @property
    def geometry_handler(self):
        """
        Lazily instantiate and return a GeometryHandler for this field.

        A geometry handler can compute distances, angles, or other geometry-based
        transformations on the given axes. If not relevant to your use case,
        this may remain unused.
        """
        if self._geometry_handler is None:
            self._geometry_handler = GeometryHandler(
                self.coordinate_system, free_axes=self._axes
            )
        return self._geometry_handler

    @property
    def AXES(self) -> List[str]:
        """
        The coordinate system axes over which this field is defined.

        Returns
        -------
        List of str
            The axes names.
        """
        return self._axes[:] # Yield a copy, non-mutable.


class ModelFieldContainer(HDF5ElementCache[str, ModelField]):
    """
     Container class that manages :py:class:`ModelField` objects within a shared HDF5 group.

     This class extends :py:class:`~pisces.io.HDF5ElementCache` to allow dictionary-like
     access to fields stored under ``manager.handle['FIELDS']``. Each field is keyed by
     a string (its name) and mapped to a :py:class:`ModelField` instance.

     Parameters
     ----------
     manager : ModelGridManager
         The manager providing global grid info and the HDF5 file handle for storing fields.
     **kwargs : dict
         Additional keyword arguments passed to :py:class:`~pisces.io.HDF5ElementCache`.

     Notes
     -----
     - The container ensures that when you retrieve or add a field, it is linked
       to the same underlying HDF5 group.
     - Fields can be referenced or created using dictionary syntax, e.g. ``my_container["density"]``.
     - This class also provides convenience methods for copying or summarizing fields.

     See Also
     --------
     HDF5ElementCache : Base caching mechanism for HDF5-backed collections.
     ModelField : Individual fields that store and retrieve array data from HDF5.
     ModelGridManager : Manages global metadata (axes, bounding box, chunking) for the grid.
     """
    def __init__(self, manager: ModelGridManager, **kwargs):
        """
        Initialize the ModelFieldContainer.

        Parameters
        ----------
        manager : ModelGridManager
            The grid manager that owns this container, providing the
            relevant HDF5 handle and coordinate system info.
        **kwargs : dict
            Passed directly to :py:class:`~pisces.io.HDF5ElementCache`
            for any specialized caching or I/O parameters.
        """
        # This __init__ simply generates a reference to the manager that got passed
        # through and then relies on the super class __init__ generate the structure.
        self._manager = manager
        super().__init__(self._manager.handle.require_group('FIELDS'), **kwargs)

    def _identify_elements_from_handle(self) -> Iterable[str]:
        # Method identifies which elements in the group are included. Its simple in
        # this case because all of the elements in the group are included.
        elements = []
        for element in self._handle.keys():
            elements.append(element)

        return elements

    def _set_element_in_handle(self, index: str, value: ModelField):
        # Sets a ``value`` in the underlying HDF5 space. Must check that
        # the ModelField we got passed is actually in the space. [Making this effectively redundant]
        if value.buffer.parent != self._handle:
            raise ValueError("The ModelField's handle is not part of this container's handle.")

        # Add the Grid's handle to the container
        self._handle[self._index_to_key(index)] = value.buffer

    def _remove_element_from_handle(self, index: str):
        del self._handle[self._index_to_key(index)]

    def _index_to_key(self, index: str) -> str:
        return index

    def _key_to_index(self, key: str) -> str:
        return key

    def load_element(self, index: str) -> ModelField:
        """
        Load a field by name (index) from the HDF5 group.

        Parameters
        ----------
        index : str
            The name of the field to load.

        Returns
        -------
        ModelField
            A newly created :py:class:`ModelField` object pointing to the dataset.

        Notes
        -----
        - This uses the :py:class:`ModelField` constructor. The field's shape,
          axes, and other attributes are taken from the on-disk HDF5 dataset.
        """
        return ModelField(self._manager,index)


    def copy_field(self, index: str, field: ModelField, overwrite: bool = False):
        """
        Make a copy of an existing :py:class:`ModelField` under a new name in this container.

        Parameters
        ----------
        index : str
            The name (key) for the copied field in this container.
        field : ModelField
            The existing field to copy. Its underlying dataset is duplicated in HDF5.
        overwrite : bool, optional
            If True, any existing field named ``index`` is removed first. Defaults to False.

        Raises
        ------
        ValueError
            If a field with the name ``index`` already exists and ``overwrite`` is False.

        Notes
        -----
        - This uses HDF5's built-in ``.copy(...)`` function, which duplicates the entire
          dataset structure. This is potentially expensive for large data sets.
        - After copying, you can retrieve the new field with ``self[index]``.
        """
        target_key = self._index_to_key(index)

        # Handle existing grid at the index
        if target_key in self._handle:
            if not overwrite:
                raise ValueError(f"A field already exists at index {index}. Use `overwrite=True` to replace it.")
            # Remove the existing group
            del self._handle[target_key]

        # Use h5py's copy method to copy the entire structure
        self._handle.copy(field.buffer, target_key)

    def add_field(self,
                  name: str,
                  /,
                  axes: Optional[Iterable[str]] = None,
                  data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
                  *,
                  overwrite: bool = False,
                  dtype: str = "f8",
                  units: str = ""):
        """
        Add a new :py:class:`ModelField` to the container.

        Parameters
        ----------
        name : str
            The name of this field. This is (or will become) the name of the dataset in the
            underlying HDF5 group.
        axes : list of str, optional
            The axes (from the underlying :py:class:`~pisces.geometry.base.CoordinateSystem`) that are
            present in this field.

            The axes are used to determine the shape of the dataset based on the ``manager``'s
            grid shape, bounding box, and other attributes.

            .. note::

                ``axes`` are **always** reordered in the method to match the ordering native to the
                coordinate system for consistency.

        data : unyt.unyt_array or numpy.ndarray, optional
            Initial data to store in the newly generated dataset. This should be a ``(...,)`` array matching
            the expected shape of the grid determined from the ``manager``. If no data is provided, then the
            array is filled with nulls.

            .. tip::

                The ``manager`` has the :py:attr:`ModelGridManager.GRID_SHAPE`, which specifies the expected
                shape for each axis. The ``data`` should then match the shape along each of the axes that
                are specified in ``axes``.


        overwrite : bool, optional
            If True, an existing field/dataset with this name is deleted and replaced. Defaults to False.
        dtype : str, optional
            The NumPy dtype to use for storing the data. Defaults to 'f8'.
        units : str or unyt.Unit, optional
            Physical units of the field. If ``data`` is a unyt array with a
            different unit, an attempt is made to convert. Defaults to '' (unitless).

        Returns
        -------
        :py:class:`ModelField`
            The newly created :py:class:`ModelField` object.
        """
        field = ModelField(self._manager,name, axes=axes, data=data, dtype=dtype, units=units, overwrite=overwrite)
        self.sync()

        return field

    def get_field_summary(self) -> Union[str, List[List[str]]]:
        """
        Generate a summary of all fields currently stored in the model.

        This summary includes:

        - Field name
        - Units of the field
        - Shape of the field data buffer
        - Axes over which the field is defined
        - Number of dimensions (ndim) of the field

        Returns
        -------
        Union[str, List[List[str]]]
            - If the `tabulate` library is installed, the summary is returned as a formatted table string.
            - If `tabulate` is not available, the summary is returned as a list of lists containing field metadata.

        Notes
        -----

        - Fields are stored within the grid manager's `FIELDS` container.
        - Each field includes metadata such as its units, shape, and axes.
        - If `tabulate` is installed, a formatted grid table is returned for better readability.
        - If `tabulate` is unavailable, the raw summary data is returned for further processing.

        Raises
        ------
        AttributeError
            If a field does not have the expected attributes like `units`, `buffer`, or `_axes`.

        See Also
        --------
        pisces.models.grids.base.ModelFieldContainer : Container for storing and managing grid fields.

        Examples
        --------
        If `tabulate` is installed, the output will look like this:

        .. code-block:: python

            +-------------+--------+-------------+------------+-------+
            | Field Name  | Units  | Shape       | Axes       | Ndim  |
            +-------------+--------+-------------+------------+-------+
            | density     | g/cm^3 | (100, 50)   | ['r', 'θ'] |   2   |
            | temperature | K      | (100, 50)   | ['r', 'θ'] |   2   |
            | pressure    | Pa     | (100, 50)   | ['r', 'θ'] |   2   |
            +-------------+--------+-------------+------------+-------+


        If `tabulate` is not installed, the method will return the following:

        .. code-block:: python

            [
                ['density', 'g/cm^3', '(100, 50)', "['r', 'θ']", '2'],
                ['temperature', 'K', '(100, 50)', "['r', 'θ']", '2'],
                ['pressure', 'Pa', '(100, 50)', "['r', 'θ']", '2']
            ]


        """
        # Import the tabulate method that we need to successfully run this.
        try:
            from tabulate import tabulate
            _use_tabulate = True
        except ImportError:
            _use_tabulate = False
            tabulate = None  #! TRICK the IDE

        # Construct the field data
        field_info = [
            [_fn, _fv.units, str(_fv.buffer.shape), str(_fv.AXES), str(_fv.buffer.ndim)] for _fn, _fv in self.items()
        ]

        if not _use_tabulate:
            return field_info
        else:
            return tabulate(field_info, headers=["Field Name", "Units", "Shape", "Axes", "Ndim"], tablefmt="grid")