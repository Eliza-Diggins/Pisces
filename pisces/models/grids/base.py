from pathlib import Path
from typing import Union, Optional, List, Iterable, Iterator, Tuple, Callable, TYPE_CHECKING, Any, Dict

import numpy as np
import unyt
from numpy.typing import NDArray, ArrayLike
from tqdm.auto import tqdm
from pisces.geometry import GeometryHandler, Symmetry
from pisces.geometry.coordinate_systems import CoordinateSystem
from pisces.io import HDF5_File_Handle, HDF5ElementCache
from pisces.models.grids.structs import BoundingBox, DomainDimensions

if TYPE_CHECKING:
    from pisces.profiles.base import Profile

ChunkIndex = Union[int, Tuple[int, ...], NDArray[np.int_]]
AxesSpecifier = Optional[Iterable[str]]
AxesMask = NDArray[np.bool_]

class ModelGridManager:
    """
    Manager class for :py:class:`ModelGrid` objects.

    This class forms the backend of all Pisces :py:class:`pisces.models.base.Model` instances and
    links the underlying HDF5 file to the grid logic.

    It provides functionality for creating and managing a multi-dimensional
    grid with specified bounding boxes, chunk shapes, and coordinate systems. It
    supports operations like computing bounding boxes and cell-centered coordinates
    for specific chunks and axes.

    Attributes
    ----------
    path : Path
        The path to the HDF5 file managed by this instance. It can point to either an existing file
        or a new file created when the `ModelGridManager` is initialized.

    handle : HDF5_File_Handle
        The file handle for the HDF5 file. This is used for reading and writing grid data and metadata.

    BBOX : BoundingBox
        The global bounding box of the grid, defining the spatial extent of the grid in physical coordinates.
        Represented as a 2D array with shape `(2, NDIM)`, where the first row is the minimum coordinates
        and the second row is the maximum coordinates along each axis.

    GRID_SHAPE : DomainDimensions
        The shape of the entire grid, given as the number of cells along each axis.
        For example, `[100, 100, 100]` represents a 3D grid with 100 cells along each axis.

    CHUNK_SHAPE : DomainDimensions
        The shape of each chunk, representing the number of cells along each axis within a chunk.
        Must evenly divide `GRID_SHAPE` along each axis.

    NCHUNKS : NDArray[np.int_]
        The number of chunks along each axis, computed as `GRID_SHAPE // CHUNK_SHAPE`.

    scale : List[str]
        The scaling for each axis, either `'linear'` or `'log'`. Determines how the bounding box
        and coordinates are interpreted and computed for each axis.

    length_unit : str
        The physical unit for lengths in the grid. For example, `'kpc'`, `'m'`, or `'cm'`.

    coordinate_system : CoordinateSystem
        The coordinate system associated with the grid (e.g., Cartesian, cylindrical, or spherical).
        Defines the axes and their names.

    _log_mask : NDArray[np.bool_]
        A boolean mask indicating which axes use logarithmic scaling. Computed based on the `scale` attribute.

    _scaled_bbox : NDArray[np.float64]
        The bounding box of the grid with scaling applied. Logarithmic axes are stored in log-transformed coordinates.

    CHUNK_SIZE : NDArray[np.float64]
        The size of each chunk in scaled coordinates along each axis. Computed as
        `(_scaled_bbox[1] - _scaled_bbox[0]) / num_chunks`.

    CELL_SIZE : NDArray[np.float64]
        The size of each cell in scaled coordinates along each axis. Computed as
        `CHUNK_SIZE / CHUNK_SHAPE`.

    _fields : ModelFieldContainer
        A container for managing the fields (data arrays) associated with the grid. Each field
        is stored in the HDF5 file under the `FIELDS` group.

    FIELDS : ModelFieldContainer
        A user-facing property for accessing the `_fields` attribute, which manages all fields
        associated with the grid. Fields can be accessed or added using dictionary-like syntax.

    Notes
    -----
    **Disk Representation**

    The grid and associated metadata are stored in an HDF5 file with the following structure:

    ```
    <root>
    ├── CSYS/
    │   ├── ...  (Coordinate system metadata)
    ├── FIELDS/
    │   ├── field_1
    │   ├── field_2
    │   ├── ...
    ├── <Attributes>
        ├── BBOX           (Bounding box: min/max coordinates of the grid)
        ├── GRID_SHAPE     (Global grid shape in number of cells along each axis)
        ├── CHUNK_SHAPE    (Chunk shape for each axis)
        ├── LUNIT          (Unit of length for spatial dimensions)
        ├── SCALE          (Scaling type for each axis: linear/log)
    ```

    - `CSYS/`: Stores the coordinate system used by the grid.
    - `FIELDS/`: Stores the data arrays (fields) associated with the grid.
    - Attributes: Global grid metadata, including the bounding box (`BBOX`), the shape of the grid
      (`GRID_SHAPE`), the size of chunks (`CHUNK_SHAPE`), and axis scaling (`SCALE`).
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
                 length_unit: str = 'kpc',
                 scale: Union[List[str], str] = 'linear'):
        """
        Initialize a ModelGridManager instance.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the HDF5 file.
        coordinate_system : CoordinateSystem, optional
            The coordinate system for the grid (required if creating a new file).
        bbox : NDArray[np.float64], optional
            The bounding box of the grid in physical units (required if creating a new file).
        grid_shape : NDArray[np.int64], optional
            The shape of the grid (required if creating a new file).
        chunk_shape : NDArray[np.int64], optional
            The shape of the chunks (required if creating a new file).
        overwrite : bool, optional
            Whether to overwrite the file if it exists. Default is False.
        length_unit : str, optional
            The unit of length for the grid. Default is "kpc".
        scale : Union[List[str], str], optional
            The scaling for each axis ('linear' or 'log'). Default is "linear".

        Raises
        ------
        ValueError
            If required parameters are missing when creating a new file or if dimensions are invalid.
        """
        # SETUP path and manage the overwrite procedure.
        self.path: Path = Path(path)

        # If file exists and overwrite is requested, delete the file
        if self.path.exists() and overwrite:
            self.path.unlink()

        # Create a new file if it doesn't exist
        if not self.path.exists():
            # Validate inputs for creating a new file
            if any(arg is None for arg in [bbox, grid_shape, coordinate_system]):
                raise ValueError(
                    f"Cannot create new ModelGridManager at {path} because not all "
                    f"of `bbox`, `grid_shape`, and `coordinate_system` were provided."
                )

            # Create the HDF5 structure
            self.handle = HDF5_File_Handle(self.path, mode='w')
            self.build_skeleton(
                self.handle,
                coordinate_system,
                bbox,
                grid_shape,
                chunk_shape,
                length_unit=length_unit,
                scale=scale
            )
            self.handle = self.handle.switch_mode('r+')
        else:
            # Open an existing file
            self.handle = HDF5_File_Handle(self.path, mode='r+')

        # Load attributes and compute derived values
        self._load_attributes()
        self._compute_secondary_attributes()

        # Load fields
        self._load_fields()

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

    def _load_attributes(self):
        """
        Load grid attributes from the HDF5 file.

        Notes
        -----
        This method loads bounding box, chunk shape, grid shape, scaling, and length unit from the file.
        """
        self.BBOX = BoundingBox(self.handle.attrs["BBOX"])
        self.CHUNK_SHAPE = DomainDimensions(self.handle.attrs["CHUNK_SHAPE"])
        self.GRID_SHAPE = DomainDimensions(self.handle.attrs["GRID_SHAPE"])
        self.scale = self.handle.attrs["SCALE"]
        self.length_unit = self.handle.attrs["LUNIT"]

        # Load the coordinate system
        self.coordinate_system = CoordinateSystem.from_file(self.handle['CSYS'], fmt='hdf5')

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
        self.NCHUNKS = self.GRID_SHAPE // self.CHUNK_SHAPE
        self.CHUNK_SIZE = (self._scaled_bbox[1, :] - self._scaled_bbox[0, :]) / self.NCHUNKS
        self.CELL_SIZE = self.CHUNK_SIZE / self.CHUNK_SHAPE

    # @@ UTILITY FUNCTIONS @@ #
    # These methods provide backend utilities for various processes
    # underlying chunk integration and operation management.

    def _generate_axes_mask(self, axes: AxesSpecifier) -> AxesMask:
        """
        Generate a mask for the specified axes in the coordinate system.

        Parameters
        ----------
        axes : AxesSpecifier
            The list of axes to generate the mask for. Must be a subset of the coordinate system's axes.

        Returns
        -------
        AxesMask
            A boolean array where `True` indicates that the axis is present in the input axes.

        Raises
        ------
        ValueError
            If any axis in `axes` is not a valid axis in the coordinate system.
        """
        if any(ax not in self.coordinate_system.AXES for ax in axes):
            raise ValueError(f"Invalid axes specified: {axes}. Valid axes are: {self.coordinate_system.AXES}.")

        return np.array([ax in axes for ax in self.coordinate_system.AXES], dtype=bool)

    def _validate_chunk_index(self, chunk_index: ChunkIndex, axes_mask: AxesMask) -> NDArray[np.int_]:
        """
        Validate the provided chunk index for the specified axes.

        Parameters
        ----------
        chunk_index : ChunkIndex
            The chunk index to validate.
        axes_mask : AxesMask
            A boolean mask indicating the axes of interest.

        Returns
        -------
        NDArray[np.int_]
            The validated chunk index.

        Raises
        ------
        ValueError
            If the chunk index is negative or exceeds the number of chunks along any axis.
        """
        chunk_index = np.array(chunk_index, dtype=int)

        for i, ci in enumerate(chunk_index):
            if ci < 0:
                raise ValueError(f"Invalid chunk index: Axis {i} has a negative index ({ci}).")
            if ci >= self.NCHUNKS[axes_mask][i]:
                raise ValueError(
                    f"Invalid chunk index: Axis {i} index ({ci}) is out of range (max: {self.NCHUNKS[axes_mask][i] - 1})."
                )

        return chunk_index

    def get_chunk_bbox(self, chunk_index: ChunkIndex, axes: Optional[AxesSpecifier] = None) -> BoundingBox:
        """
        Compute the bounding box for a specific chunk along the specified axes.

        Parameters
        ----------
        chunk_index : ChunkIndex
            The index of the chunk.
        axes : AxesSpecifier
            The axes for which the bounding box is computed.

        Returns
        -------
        BoundingBox
            The bounding box of the chunk in scaled coordinates.

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        if axes is None:
            axes = self.coordinate_system.AXES
        axes_mask = self._generate_axes_mask(axes)
        chunk_index = self._validate_chunk_index(chunk_index, axes_mask)

        left = self._scaled_bbox[0, axes_mask] + chunk_index * self.CHUNK_SIZE[axes_mask]
        right = left + self.CHUNK_SIZE[axes_mask]

        return BoundingBox(np.stack((left, right), axis=-1))

    def get_chunk_mask(self, chunk_index: ChunkIndex, axes: Optional[AxesSpecifier] = None) -> List[slice]:
        """
        Compute the slice indices for a specific chunk along the specified axes.

        Parameters
        ----------
        chunk_index : ChunkIndex
            The index of the chunk.
        axes : AxesSpecifier
            The axes for which the slice indices are computed.

        Returns
        -------
        List[slice]
            A list of slice objects representing the chunk indices.

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        if axes is None:
            axes = self.coordinate_system.AXES

        axes_mask = self._generate_axes_mask(axes)
        chunk_index = self._validate_chunk_index(chunk_index, axes_mask)

        start = chunk_index * self.CHUNK_SHAPE[axes_mask]
        end = (chunk_index + 1) * self.CHUNK_SHAPE[axes_mask]

        return [slice(s, e) for s, e in zip(start, end)]

    def get_coordinates(
            self,
            chunk_index: Optional[ChunkIndex] = None,
            axes: Optional[AxesSpecifier] = None,
    ) -> NDArray[np.float64]:
        """
        Compute the cell-centered coordinates for the grid or a specific chunk.

        Parameters
        ----------
        chunk_index : Optional[ChunkIndex], optional
            The index of the chunk for which coordinates are computed. If `None`, compute for the entire grid.
        axes : Optional[AxesSpecifier], optional
            The axes for which coordinates are computed. If `None`, use all axes.

        Returns
        -------
        NDArray[np.float64]
            An array of cell-centered coordinates with shape `(*GRID_SHAPE, len(axes))`.

        Raises
        ------
        ValueError
            If invalid axes or chunk indices are provided.
        """
        if axes is None:
            axes = self.coordinate_system.AXES

        axes_mask = self._generate_axes_mask(axes)

        if chunk_index is not None:
            chunk_index = self._validate_chunk_index(chunk_index, axes_mask)
            cbbox = self.get_chunk_bbox(chunk_index, axes)
            cshape = self.CHUNK_SHAPE[axes_mask]
        else:
            cbbox = self.BBOX[:, axes_mask]
            cshape = self.GRID_SHAPE[axes_mask]

        log_mask = self._log_mask[axes_mask]
        cell_size = self.CELL_SIZE[axes_mask]

        slices = [
            slice(cbbox[0, i] + 0.5 * cell_size[i], cbbox[1, i] - 0.5 * cell_size[i], cshape[i] * 1j)
            for i in range(len(axes))
        ]

        coordinates = np.moveaxis(np.mgrid[*slices], 0, -1)
        coordinates[..., log_mask] = 10**coordinates[..., log_mask]
        axes_indices = [[ax for ax in self.coordinate_system.AXES if ax in axes].index(_ax) for _ax in axes]
        return coordinates[..., axes_indices]

    def iter_chunks(self, axes: Optional[List[str]] = None) -> Iterator[NDArray[np.int_]]:
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
        if axes is None:
            axes = self.coordinate_system.AXES

        mask = self._generate_axes_mask(axes)
        chunk_indices = np.indices(self.NCHUNKS[mask], dtype=int).reshape(len(self.NCHUNKS[mask]), -1).T

        for chunk_index in chunk_indices:
            yield chunk_index

    def set_in_chunks(
            self,
            output_field: str,
            input_fields: List[str],
            function: Callable[..., NDArray[np.float64]],
            axes: Optional[List[str]] = None,
            **kwargs,
    ):
        """
        Compute and set a field's values in chunks by applying a function to other fields.

        Parameters
        ----------
        output_field : str
            The name of the output field to create or update.
        input_fields : List[str]
            Names of the input fields required by the function.
        function : Callable[..., NDArray[np.float64]]
            A function that computes the output field values from the input fields.
        axes : Optional[List[str]], optional
            The axes over which to iterate. Defaults to the axes of the output field.
        **kwargs : dict
            Additional arguments passed to the function.

        Raises
        ------
        ValueError
            If the required input or output fields are missing or invalid.
        """
        # Validate output field
        if output_field not in self.FIELDS:
            raise ValueError(f"Output field '{output_field}' is not a known field.")
        _outfield = self.FIELDS[output_field]

        # Validate axes
        if axes is None:
            axes = _outfield.AXES
        axes_mask = self._generate_axes_mask(axes=axes)

        # Validate input fields and prepare axis mapping
        chunk_masks = {}
        for input_field in input_fields:
            if input_field not in self.FIELDS:
                raise ValueError(f"Input field '{input_field}' is not a known field.")
            input_axes = self.FIELDS[input_field].AXES

            # Ensure input field axes are compatible with iteration axes
            if not all(ax in axes for ax in input_axes):
                raise ValueError(
                    f"Input field '{input_field}' contains invalid axes which are not in {axes}. "
                    "This prevents efficient chunking from occurring."
                )

            # Generate chunk mask for input field based on axes
            input_axes_mask = self._generate_axes_mask(input_axes)
            chunk_masks[input_field] = input_axes_mask

        # Initialize progress bar
        pbar = tqdm(
            desc=f"Set In Chunks: {output_field}",
            total=int(np.prod(self.NCHUNKS[axes_mask])),
        )

        # Perform chunk-wise computation
        for chunk_index in self.iter_chunks(axes):
            # Get the chunk mask for the output field
            chunk_mask = self.get_chunk_mask(chunk_index, axes=axes)

            # Prepare input field chunks
            chunk_data = []
            for input_field in input_fields:
                input_chunk_index = chunk_index[chunk_masks[input_field]]
                input_chunk_mask = self.get_chunk_mask(input_chunk_index, self.FIELDS[input_field].AXES)
                chunk_data.append(self.FIELDS[input_field][tuple(input_chunk_mask)])

            # Compute the result for this chunk
            result = function(*chunk_data, **kwargs)

            # Write the result to the corresponding chunk of the output field
            _outfield[tuple(chunk_mask)] = result

            # Update progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

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
        Add a field to the grid by evaluating a function at the grid points.

        Parameters
        ----------
        function : Callable
            A function that computes the field values. Must accept coordinate arrays as inputs.
        field_name : str
            The name of the field to be added.
        axes : Optional[List[str]], optional
            The axes along which the field is defined. If `None`, use all axes.
        chunking : bool, optional
            If `True`, evaluate the function in chunks. Default is `False`.
        units : Optional[str], optional
            The units of the field. Default is `None`.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.

        Raises
        ------
        ValueError
            If the function, axes, or other parameters are invalid.
        """
        self.FIELDS.add_field(
            field_name,
            axes=axes,
            units=units,
            dtype=dtype,
            overwrite=overwrite,
        )

        _field = self.FIELDS[field_name]
        if chunking:
            pbar = tqdm(desc=f"Adding field '{field_name}'",
                        total = int(np.prod(self.NCHUNKS[axes])))
            for chunk_index in self.iter_chunks(axes):
                chunk_mask = self.get_chunk_mask(chunk_index, axes)
                chunk_coordinates = self.get_coordinates(chunk_index, axes)
                _field[*chunk_mask] = function(*np.moveaxis(chunk_coordinates, -1, 0), **kwargs)
                pbar.update(1)
            pbar.close()
        else:
            _field[...] = function(*np.moveaxis(self.get_coordinates(axes=axes), -1, 0), **kwargs)

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
        Add a field to the grid by evaluating a profile.

        Parameters
        ----------
        profile : Profile
            The profile used to compute the field values. Must have an `.axes` attribute.
        field_name : str
            The name of the field to be added.
        chunking : bool, optional
            If `True`, evaluate the profile in chunks to handle large grids. Default is `False`.
        units : Optional[str], optional
            The units of the field. If `None`, defaults to the units of the profile.
        dtype : str, optional
            The data type of the field. Default is "f8".
        overwrite : bool, optional
            If `True`, overwrite an existing field with the same name. Default is `False`.
        **kwargs : dict
            Additional keyword arguments passed to the profile evaluation function.

        Raises
        ------
        ValueError
            If the profile's axes are incompatible with the grid's coordinate system.
        """
        # Validate that the profile's axes align with the grid's coordinate system
        if not all(ax in self.coordinate_system.AXES for ax in profile.axes):
            raise ValueError(
                f"Profile axes {profile.axes} are incompatible with the grid's coordinate system axes: {self.coordinate_system.AXES}."
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
            axes=profile.axes,
            chunking=chunking,
            units=units,
            dtype=dtype,
            overwrite=overwrite,
            **kwargs,
        )

    # @@ BUILDERS @@ #
    @classmethod
    def build_skeleton(cls,
                       handle: HDF5_File_Handle,
                       coordinate_system: CoordinateSystem,
                       bbox: NDArray[np.floating],
                       grid_shape: NDArray[np.int_],
                       chunk_shape: NDArray[np.int_],
                       length_unit: str = 'kpc',
                       scale: Union[List[str] | str] = 'linear',
                       ) -> HDF5_File_Handle:
        # ADDING the coordinate system to the HDF5 handle.
        coordinate_system.to_file(handle.require_group('CSYS'), fmt='hdf5')

        # VALIDATE bbox, grid_shape, chunk_shape.
        if chunk_shape is None:
            chunk_shape = grid_shape
        bbox, grid_shape, chunk_shape = BoundingBox(bbox), DomainDimensions(grid_shape), DomainDimensions(
            chunk_shape)
        _dimension_check_set = [bbox.shape[-1], len(grid_shape), len(chunk_shape), coordinate_system.NDIM]

        # Check dimensions are consistent.
        if len(set(_dimension_check_set)) != 1:
            raise ValueError(
                f"Detected inconsistent dimensions while building skeleton: dimensions for bbox, grid_shape,"
                f" chunk_shape, and coordinate system were {_dimension_check_set} respectively.")

        # Check that the chunks fit
        if np.any(grid_shape % chunk_shape != 0):
            raise ValueError(f"Grid shape {grid_shape} must be divisible by chunk shape {chunk_shape}.")

        # VALIDATE scale
        if isinstance(scale, str):
            scale = [scale] * coordinate_system.NDIM
        else:
            if len(scale) != coordinate_system.NDIM:
                raise ValueError(f"Scale {scale} must be specified for all dimensions.")

        if any(k not in ['linear', 'log'] for k in scale):
            raise ValueError(f"Scale {scale} must be linear or log.")

        # VALIDATE length_unit
        length_unit = str(length_unit)

        # WRITE parameters to disk
        handle.attrs['LUNIT'] = length_unit
        handle.attrs['SCALE'] = scale
        handle.attrs['CHUNK_SHAPE'] = chunk_shape
        handle.attrs['BBOX'] = bbox
        handle.attrs['GRID_SHAPE'] = grid_shape

        return handle

    @property
    def FIELDS(self) -> 'ModelFieldContainer':
        """
        Provides access to the fields container.

        This container allows users to interact with the fields (data arrays) associated with the grid.
        Fields can be accessed, modified, or added using dictionary-like syntax.

        Returns
        -------
        ModelFieldContainer
            The container for managing the fields in the grid.

        """
        if self._fields is None:
            raise AttributeError("Fields container has not been initialized. Ensure the grid manager is correctly initialized.")
        return self._fields

class ModelField(unyt.unyt_array):
    """
    A ModelField is effectively an unyt array with some notion of its own geometry vis-a-vis a
    connection to a ModelGridManager.

    These classes cannot be instantiated without them.
    """

    def __new__(cls,
                manager: ModelGridManager,
                name: str,
                /,
                axes: Optional[List[str]] = None,
                data: Optional[Union[unyt.unyt_array,np.ndarray]] = None,
                *,
                overwrite: bool = False,
                dtype: str = 'f8',
                units: str = ''):
        # Load the dataset from the skeleton constructor.
        # This will either generate the structure or return a
        # pre-existing one.
        dataset = cls.build_skeleton(manager,name,axes,data,overwrite=overwrite,dtype=dtype,units=units)

        # Obtain the units and axes from the HDF5 group.
        _units,_axes = dataset.attrs.get('units',units),dataset.attrs['axes']

        # Create the object.
        obj = super().__new__(cls, [], units=units)
        obj._name = name
        obj.units = _units
        obj.dtype = dataset.dtype
        obj.buffer = dataset
        obj._manager = manager
        obj._axes = axes
        obj._gh = None

        return obj

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
        units: str = "",
    ):
        # Setup the handler and manage the overwriting procedure.
        handle = manager.handle.require_group("FIELDS")
        if overwrite and name in handle:
            del handle[name]

        # Return existing dataset if it exists
        if name in handle:
            return handle[name]

        # ---------------------------------- #
        # Parameter validation and coercion  #
        # ---------------------------------- #
        # Manage the axes. These must be valid axes in the coordinate
        # system.
        _coordinate_system = manager.coordinate_system
        if axes is None:
            axes = _coordinate_system.AXES[:]
        if any(ax not in _coordinate_system.AXES for ax in axes):
            raise ValueError(f"The following axes are not recognized for the {_coordinate_system.__class__.__name__}"
                             f" coordinate system: {[ax for ax in axes if ax not in _coordinate_system.AXES]}")

        # Construct the expected shape and ensure that, if data is provided
        # it has the correct shape.
        axes_indices = np.array([_coordinate_system.AXES.index(ax) for ax in axes],dtype='int')
        shape = manager.GRID_SHAPE[axes_indices]

        if data is not None:
            if not np.array_equal(data.shape,shape):
                raise ValueError(f"Expected shape {shape} but received shape {data.shape}.")

        # Convert units and dtype if necessary
            if isinstance(data, unyt.unyt_array):
                data_units = str(data.units)
                if units:
                    try:
                        data = data.to_value(units)
                    except Exception as e:
                        raise ValueError(
                            f"Inconsistent units: provided data has units '{data_units}', "
                            f"but specified units are '{units}'."
                        ) from e
                else:
                    units = data_units

        # Create the dataset
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
        inputs = tuple(x.view(unyt.unyt_array) if isinstance(x, ModelField) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, unyt.unyt_array):
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
        Retrieve a slice of the field's data.

        Parameters
        ----------
        key : slice or int
            The indices to retrieve.

        Returns
        -------
        unyt.unyt_array
            The retrieved data with units.

        Notes
        -----
        The retrieved data is returned as a `unyt_array`, preserving the field's
        physical units.
        """
        return unyt.unyt_array(self.buffer[key], units=self.units)

    @property
    def coordinate_system(self):
        return self._manager.coordinate_system

    @property
    def geometry_handler(self):
        if self._gh is None:
            # Construct the handler from the axes
            _sym_axes = [ax for ax in self.coordinate_system.AXES if ax not in self._axes]
            _sym = Symmetry(_sym_axes,self.coordinate_system)
            self._gh = GeometryHandler(self.coordinate_system,_sym)

        return self._gh

class ModelFieldContainer(HDF5ElementCache[str, ModelField]):

    def __init__(self, manager: ModelGridManager, **kwargs):
        # CREATE reference to the `ModelGridManager`.
        self._manager = manager
        super().__init__(self._manager.handle.require_group('FIELDS'), **kwargs)

    def _identify_elements_from_handle(self) -> Iterable[str]:
        elements = []
        for element in self._handle.keys():
            elements.append(element)

        return elements

    def _set_element_in_handle(self, index: str, value: ModelField):
        # Check if the Grid's handle is part of this container
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
        return ModelField(self._manager,index)


    def copy_field(self, index: str, field: ModelField, overwrite: bool = False):
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
        field = ModelField(self._manager,name, axes=axes, data=data, dtype=dtype, units=units, overwrite=overwrite)
        self.sync()

        return field


