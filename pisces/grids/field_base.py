from typing import Union, Iterable, TYPE_CHECKING, Tuple, Optional

import h5py
import numpy as np
import unyt

from pisces.io.hdf5 import HDF5ElementCache

if TYPE_CHECKING:
    from pisces.grids.grid_base import Grid


class Field(unyt.unyt_array):
    """
    Represents a physical field on a grid, stored as an HDF5-backed dataset.

    The `Field` class extends `unyt_array` to support unit-aware arithmetic operations
    while maintaining efficient storage and retrieval of data using HDF5. Each `Field`
    instance corresponds to a physical quantity associated with a specific `Grid`.

    Attributes
    ----------
    name : str
        The name of the field.
    units : :py:class:`unyt.Unit`
        The units of the field.
    dtype : numpy.dtype
        The data type of the field.
    buffer : h5py.Dataset
        The HDF5 dataset backing this field.
    _grid : :py:class:`grids.grid_base.Grid`
        The `Grid` object associated with this field.

    See Also
    --------
    Grid : Represents a single grid in the hierarchy.
    FieldContainer : Manages multiple fields associated with a grid.

    Notes
    -----
    - Fields support lazy loading and writing of data slices to minimize memory overhead.
    - Metadata (e.g., units) is automatically synchronized with the underlying HDF5 dataset.

    Examples
    --------
    .. code-block:: python

        # Creating a Field associated with a grid
        field = Field("temperature", grid, units="K", dtype="f8", overwrite=True)

        # Writing data to the Field
        field[:] = np.random.rand(*grid.GS)

        # Reading data from the Field
        data = field[:]
    """

    def __new__(cls, name: str, grid: "Grid", overwrite: bool = False, **kwargs):
        """
        Initialize or load a Field instance.

        Parameters
        ----------
        name : str
            Name of the field dataset in the HDF5 file.
        grid : :py:class:`grids.grid_base.Grid`
            The grid instance this field is associated with.
        overwrite : bool, optional
            If True, overwrites an existing dataset with the same name.
        **kwargs :
            Additional parameters for dataset creation (e.g., `units`, `dtype`, `data`).

        Returns
        -------
        Field
            An initialized `Field` object.
        """
        # Create or retrieve the dataset
        dataset = cls.construct_dataset(name, grid, overwrite=overwrite, **kwargs)

        # Initialize the unyt_array with the dataset's shape, dtype, and units
        units = dataset.attrs.get("units", "")
        obj = super().__new__(cls, [], units=units)
        obj.name = name
        obj.units = units
        obj.dtype = dataset.dtype
        obj.buffer = dataset
        obj._grid = grid

        return obj

    @classmethod
    def construct_dataset(cls, name: str, grid: "Grid", overwrite: bool = False, **kwargs) -> h5py.Dataset:
        """
        Create or retrieve the HDF5 dataset for the field.

        If the dataset already exists in the HDF5 group, it is retrieved unless
        `overwrite=True`. If it does not exist, a new dataset is created based
        on the grid's shape and provided parameters.

        Parameters
        ----------
        name : str
            Name of the dataset in the HDF5 group.
        grid : :py:class:`grids.grid_base.Grid`
            The grid instance this field is associated with.
        overwrite : bool, optional
            If True, deletes the existing dataset before creating a new one.
        **kwargs : dict
            Parameters for dataset creation:
            - `data` (array-like): Initial data for the dataset. May be a `unyt_array`.
            - `units` (str): Units of the field.
            - `dtype` (str): Data type of the dataset.

        Returns
        -------
        h5py.Dataset
            The created or retrieved dataset.

        Raises
        ------
        ValueError
            If the provided data shape does not match the grid shape.
            If the units of `data` (if provided) are incompatible with the specified units.

        See Also
        --------
        :py:class:`Field`
            Represents a single physical field on a grid.

        Notes
        -----
        This method ensures the dataset exists and conforms to the specified parameters.
        It handles unit compatibility and assigns appropriate metadata.
        """
        handle = grid.handle
        grid_shape = tuple(grid.GS)

        # Overwrite the dataset if requested
        if overwrite and name in handle:
            del handle[name]

        # Retrieve the existing dataset or create a new one
        if name in handle:
            dataset = handle[name]
        else:
            # Parse dataset creation parameters
            data = kwargs.get("data", None)
            dtype = kwargs.get("dtype", "f8")
            units = kwargs.get("units", "")

            # Handle units for provided data
            if isinstance(data, unyt.unyt_array):
                data_units = str(data.units)
                if units and not unyt.Unit(units).is_compatible(data_units):
                    raise ValueError(
                        f"Inconsistent units: provided data has units '{data_units}' "
                        f"but specified units are '{units}'."
                    )
                # Use the units from the data if not explicitly provided
                units = units or data_units
                data = data.d  # Extract raw array for HDF5 storage

            # Assign default units if none are provided
            if not units:
                units = ""

            # Validate provided data shape
            if data is not None and data.shape != grid_shape:
                raise ValueError(f"Provided data shape {data.shape} does not match grid shape {grid_shape}.")

            # Create the dataset
            dataset = handle.create_dataset(name, shape=grid_shape, dtype=dtype, data=data)

            # Set dataset attributes
            dataset.attrs["units"] = units

        return dataset

    def inspect_field_metadata(self) -> dict:
        """
        Inspect metadata of the field.

        Returns
        -------
        dict
            Metadata including units, shape, and dtype.

        Examples
        --------
        .. code-block:: python

            metadata = field.inspect_field_metadata()
            print(metadata)
        """
        return {
            "name": self.name,
            "units": self.units,
            "dtype": self.dtype,
            "shape": self.buffer.shape,
        }

    def resize_field(self, new_shape: Tuple[int, ...]):
        """
        Resize the field dataset.

        Parameters
        ----------
        new_shape : Tuple[int, ...]
            New shape for the dataset.

        Notes
        -----
        Resizing is only allowed if the HDF5 dataset supports resizing.

        Examples
        --------
        .. code-block:: python

            field.resize_field((100, 200))
        """
        self.buffer.resize(new_shape)

    def clear_field(self):
        """
        Clear the field's data by setting all values to zero.

        Examples
        --------
        .. code-block:: python

            field.clear_field()
        """
        self.buffer[:] = 0

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

        if not value.units.is_compatible(self.units):
            raise ValueError(
                f"Units of value '{value.units}' are not compatible with field '{self.name}' units '{self.units}'.")

        # Write the data to the HDF5 dataset
        self.buffer[key] = value.to(self.units).d

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
        inputs = tuple(x.view(unyt.unyt_array) if isinstance(x, Field) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, unyt.unyt_array):
            result.units = self.units

        return result

    def __repr__(self) -> str:
        return f"<Field {self.grid}/{self.name}, units={self.units}>"

    def __str__(self) -> str:
        return f"<Field {self.grid}/{self.name}, units={self.units}>"

    @property
    def grid(self) -> "Grid":
        """
        Get the grid this field is associated with.

        Returns
        -------
        grids.grid_base.Grid
            The grid object associated with this field.

        Notes
        -----
        Provides a direct link to the parent grid, allowing for hierarchical
        navigation between fields and grids.
        """
        return self._grid


class FieldContainer(HDF5ElementCache[str, Field]):
    """
    Container for managing multiple fields associated with a grid.

    See Also
    --------
    Field : Represents individual physical fields.
    """

    def __init__(self, grid: 'Grid', **kwargs):
        # Create the grid reference.
        self._grid = grid

        # Initialize via standard route.
        super().__init__(self._grid.handle, **kwargs)

    def _identify_elements_from_handle(self) -> Iterable[str]:
        elements = []
        for element in self._handle.keys():
            elements.append(element)

        return elements

    def _set_element_in_handle(self, index: str, value: Field):
        """
        Add a Grid to the container, ensuring its handle is part of the container.

        Parameters
        ----------
        index : str
            The index at which to add the Grid.
        value : Grid
            The Grid instance to add.

        Raises
        ------
        ValueError
            If the Grid's handle is not part of the container's handle.
        """
        # Check if the Grid's handle is part of this container
        if value.buffer.parent != self._handle:
            raise ValueError("The Field's handle is not part of this container's handle.")

        # Add the Grid's handle to the container
        self._handle[self._index_to_key(index)] = value.buffer

    def _remove_element_from_handle(self, index: str):
        del self._handle[self._index_to_key(index)]

    def _index_to_key(self, index: str) -> str:
        return index

    def _key_to_index(self, key: str) -> str:
        return key

    def load_element(self, index: str) -> Field:
        return Field(index, self._grid)

    def copy_field(self, index: str, field: Field, overwrite: bool = False):
        target_key = self._index_to_key(index)

        # Handle existing grid at the index
        if target_key in self._handle:
            if not overwrite:
                raise ValueError(f"A grid already exists at index {index}. Use `overwrite=True` to replace it.")
            # Remove the existing group
            del self._handle[target_key]

        # Use h5py's copy method to copy the entire structure
        self._handle.copy(field.buffer, target_key)


def add_field(self, name: str, data: Optional[np.ndarray] = None, units: str = "", dtype: str = "f8",
              overwrite: bool = False) -> Field:
    """
    Add a new field to the container.

    Parameters
    ----------
    name : str
        The name of the field to add.
    data : np.ndarray, optional
        Initial data for the field. If not provided, the field will be created as an empty dataset.
    units : str, optional
        Units of the field, specified as a string compatible with `unyt`.
    dtype : str, optional
        Data type of the field. Default is "f8" (64-bit floating point).
    overwrite : bool, optional
        If True, overwrites any existing field with the same name. Default is False.

    Returns
    -------
    Field
        The newly created `Field` object.

    Raises
    ------
    ValueError
        If a field with the same name already exists and `overwrite` is False.
    ValueError
        If the provided data shape does not match the grid's dimensions.

    Examples
    --------
    .. code-block:: python

        # Add a new field with random data
        grid = Grid(...)  # Initialize a grid
        container = FieldContainer(grid)
        data = np.random.rand(*grid.GS)
        field = container.add_field("density", data=data, units="g/cm**3")

        # Add an empty field
        empty_field = container.add_field("temperature", units="K")

        # Overwrite an existing field
        updated_field = container.add_field("density", data=new_data, overwrite=True)
    """
    # Check if a field with the same name already exists
    if name in self._handle:
        if not overwrite:
            raise ValueError(f"Field '{name}' already exists. Use `overwrite=True` to replace it.")
        del self._handle[name]  # Remove existing field if overwriting

    # Validate the shape of the provided data
    if data is not None and data.shape != tuple(self._grid.GS):
        raise ValueError(f"Provided data shape {data.shape} does not match grid shape {tuple(self._grid.GS)}.")

    # Create the new field and add it to the container
    field = Field(name, self._grid, data=data, units=units, dtype=dtype, overwrite=overwrite)
    self.sync()  # Synchronize the container with the HDF5 structure

    return field
