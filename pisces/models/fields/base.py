import unyt
from typing import Tuple, TYPE_CHECKING, Optional, Union, List, Set, Iterable
import numpy as np
from pisces.geometry import GeometryHandler, Symmetry
from pisces.io.hdf5 import HDF5ElementCache


if TYPE_CHECKING:
    from pisces.models.base import Model

class ModelField(unyt.unyt_array):
    """
    Represents a grid field in a :py:class:`Model` instance. This class creates or references fields
    stored as datasets in the HDF5 file under `/FIELDS/`.
    """

    def __new__(cls, model, name: str, overwrite: bool = False, **kwargs):
        """
        Creates or retrieves a field dataset in the HDF5 file.

        Parameters
        ----------
        model : Model
            The parent model instance to which the field belongs.
        name : str
            Name of the field.
        overwrite : bool, optional
            Whether to overwrite an existing field with the same name, by default False.

        Returns
        -------

            The HDF5 dataset associated with the field.
        """
        # Create or retrieve the dataset
        dataset = cls.construct_skeleton(model,name, overwrite=overwrite, **kwargs)

        # Load the units and axes.
        try:
            units = dataset.attrs.get("units", "")
            axes = set(dataset.attrs['axes'])
        except Exception as e:
            raise ValueError(f"Failed to initialize units / axes for field {name} of {model}: {e}.") from e

        # Create the object.
        obj = super().__new__(cls, [], units=units)
        obj._name = name
        obj.units = units
        obj.dtype = dataset.dtype
        obj.buffer = dataset
        obj._model = model
        obj._axes = axes
        obj._gh = None

        return obj

    @classmethod
    def construct_skeleton(
        cls,
        model: "Model",
        name: str,
        *,
        axes: List[str],
        overwrite: bool = False,
        data: Optional[Union[unyt.unyt_array, np.ndarray]] = None,
        dtype: str = "f8",
        units: str = "",
        shape: Optional[tuple] = None,
    ):
        """
        Creates the field dataset in the HDF5 file.

        Parameters
        ----------
        model : Model
            The parent model instance.
        name : str
            Name of the field.
        axes : list of str
            Axes associated with the field (e.g., ['X', 'Y', 'Z']).
        overwrite : bool, optional
            Whether to overwrite an existing field with the same name, by default False.
        data : unyt.unyt_array or np.ndarray, optional
            Initial data for the field, by default None.
        dtype : str, optional
            Data type for the field, by default 'f8'.
        units : str, optional
            Units for the field data, by default an empty string.
        shape : tuple, optional
            Shape of the field, by default None.

        Returns
        -------
        h5py.Dataset
            The HDF5 dataset for the field.

        Raises
        ------
        ValueError
            If axes are not compatible with the model's coordinate system or
            if neither `data` nor `shape` are provided.
        """
        handle = model.handle["FIELDS"]

        # Handle overwrite
        if overwrite and name in handle:
            del handle[name]

        # Return existing dataset if it exists
        if name in handle:
            return handle[name]

        # Validate axes
        axes = set(axes)
        if not axes.issubset(model.coordinate_system.AXES):
            raise ValueError(
                f"Cannot construct field '{name}' for {model} with axes {axes}. "
                f"Model uses coordinate system '{model.coordinate_system.__class__.__name__}' "
                f"with axes {model.coordinate_system.AXES}."
            )

        # Validate data and shape
        if data is None and shape is None:
            raise ValueError(
                f"Cannot create field '{name}' because neither `data` nor `shape` were provided."
            )
        if data is not None:
            if shape is None:
                shape = data.shape
            elif data.shape != shape:
                raise ValueError(
                    f"Inconsistent shapes: provided data has shape {data.shape}, "
                    f"but specified shape is {shape}."
                )

        # Convert units and dtype if necessary
        if isinstance(data, unyt.unyt_array):
            data_units = str(data.units)
            if units and not unyt.Unit(units).is_compatible(data_units):
                raise ValueError(
                    f"Inconsistent units: provided data has units '{data_units}', "
                    f"but specified units are '{units}'."
                )
            # Use data's units if none are specified
            units = units or data_units
            data = data.to_value(units)

        # Validate shape against axes
        if len(shape) < len(axes):
            raise ValueError(
                f"Field shape {shape} is incompatible with the number of specified axes {axes}."
            )

        # Create the dataset
        dataset = handle.create_dataset(name, shape=shape, data=data, dtype=dtype)

        # Set attributes
        dataset.attrs["units"] = units
        dataset.attrs["axes"] = list(axes)

        return dataset

    @property
    def geometry_handler(self) -> GeometryHandler:
        if self._gh is None:
            _s_axes = {ax for ax in self._model.coordinate_system.AXES if ax not in self._axes}
            self._gh = GeometryHandler(self._model.coordinate_system,Symmetry(_s_axes,self._model.coordinate_system))

        return self._gh

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
        self.buffer[key] = value.to_value(self.units)

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
        inputs = tuple(x.view(unyt.unyt_array) if isinstance(x, ModelField) else x for x in inputs)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if isinstance(result, unyt.unyt_array):
            result.units = self.units

        return result

class ModelFieldContainer(HDF5ElementCache[str, ModelField]):
    """
    Container for managing multiple fields associated with a grid.

    See Also
    --------
    ModelField : Represents individual physical fields.
    """

    def __init__(self, model: 'Model', **kwargs):
        # Create the grid reference.
        self._model = model

        # Initialize via standard route.
        super().__init__(self._model.handle['FIELDS'], **kwargs)

    def _identify_elements_from_handle(self) -> Iterable[str]:
        elements = []
        for element in self._handle.keys():
            elements.append(element)

        return elements

    def _set_element_in_handle(self, index: str, value: ModelField):
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
        return ModelField(self._model,index)


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


    def add_field(self, name: str, data: Optional[np.ndarray] = None, units: str = "", dtype: str = "f8",
                  overwrite: bool = False) -> ModelField:
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
        ModelField
            The newly created `ModelField` object.

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
            container = ModelFieldContainer(grid)
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
                raise ValueError(f"ModelField '{name}' already exists. Use `overwrite=True` to replace it.")
            del self._handle[name]  # Remove existing field if overwriting


        # Create the new field and add it to the container
        field = ModelField(self._model, name, data=data, units=units, dtype=dtype, overwrite=overwrite)
        self.sync()  # Synchronize the container with the HDF5 structure

        return field