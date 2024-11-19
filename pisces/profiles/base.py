from abc import ABCMeta
from typing import TYPE_CHECKING, Callable, Union, List, Dict, Any
from numpy.typing import ArrayLike
import h5py
import unyt
import dill
import numpy as np
from pisces.profiles.registries import _DEFAULT_PROFILE_REGISTRY
from pisces.profiles._typing import ProfileType
from pisces.utilities.math import function_partial_derivative

if TYPE_CHECKING:
    from pisces.profiles.registries import ProfileRegistry


class ProfileMeta(ABCMeta):
    """
    Metaclass for managing profile classes and automatically registering concrete implementations.

    Attributes
    ----------
    REGISTRY : ProfileRegistry
        A registry for storing profile class implementations.
    """
    REGISTRY: 'ProfileRegistry' = _DEFAULT_PROFILE_REGISTRY
    _IGNORED = ['Profile']
    def __init__(cls, name, bases, clsdict):
        """
        Initialize a ProfileMeta class instance and register concrete subclasses.

        Parameters
        ----------
        name : str
            The name of the class being created.
        bases : tuple
            The base classes of the class being created.
        clsdict : dict
            Dictionary containing the class attributes and methods.

        Notes
        -----
        - Registers only fully implemented subclasses (i.e., classes with no abstract methods).
        - Skips registration if the class is already in the registry.
        """
        super().__init__(name, bases, clsdict)

        # Register only if not abstract and not already registered
        if (name not in cls.REGISTRY.registry) and (name not in cls._IGNORED):
            # Check for abstract methods
            if not getattr(cls, '__abstractmethods__', None):
                cls.REGISTRY.register(cls)
            else:
                # Abstract class, skipped for registration
                pass


class Profile(metaclass=ProfileMeta):
    """
    A class representing a profile with a function, units, and various parameters.

    Parameters
    ----------
    function : Callable
        A callable function defining the profile.
    axes : List[str]
        List of axis names the profile function depends on.
    units : Union[str, unyt.Unit], optional
        The units of the profile, default is an empty string.
    kwargs : dict
        Additional parameters for the profile function.
    """

    def __init__(self,
                 function: ProfileType,
                 axes: List[str],
                 /,
                 units: Union[str,unyt.Unit] = '',
                 **kwargs):
        # Initialize the parameters provided in kwargs.
        self.parameters: Dict[str,Any] = kwargs
        self.axes: List[str] = axes
        self.units: unyt.Unit = unyt.Unit(units)

        # Generate function attribute to store the connection to the
        # callable.
        self._function: ProfileType = function

    def __call__(self, *args) -> Any:
        """
        Call the profile function with arguments and profile parameters.
        """
        return self._function(*args, **self.parameters)

    def __repr__(self) -> str:
        """
        Return a string representation of the Profile instance.

        Returns
        -------
        str
            A string representation showing the class name and parameters.
        """
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"<{self.__class__.__name__}({param_str})>"

    def __str__(self):
        return self.__repr__()

    def _do_operation(self, other: Union["Profile", float, int], op: Callable, validate=False) -> 'Profile':
        """
        Internal method to perform operations between Profile instances or scalars.

        Parameters
        ----------
        other : Union[Profile, float, int]
            The other profile or scalar to combine with this profile.
        op : Callable
            The operation to perform (e.g., addition, subtraction).
        validate : bool, optional
            If True, validates unit compatibility.
        """
        if isinstance(other, Profile):
            # The other instance is another profile. We need to validate and then
            # coerce arguments for combination.
            if validate and isinstance(other, Profile):
                if self.units != other.units:
                    raise ValueError(f"Unit mismatch: {self.units} (self) vs {other.units} (other)")

            # Setup shared attributes.
            sf, of = self._function, other._function
            axes = self.axes + [ax for ax in other.axes if ax not in self.axes]

            # Determine slices for each function based on the axes
            self_slices = [axes.index(ax) for ax in self.axes]
            other_slices = [axes.index(ax) for ax in other.axes]

            # Merge parameters from self and other
            combined_parameters = self.parameters.copy()  # Start with self's parameters
            for k, v in other.parameters.items():
                if k in combined_parameters:
                    # Handle conflicting parameter names here, if needed
                    raise ValueError(f"Parameter conflict: {k} exists in both profiles.")
                combined_parameters[k] = v

            def combined_function(*coords):
                # Slice coordinates according to self's and other's axes
                self_coords = [coords[i] for i in self_slices]
                other_coords = [coords[i] for i in other_slices]

                # Apply each function and combine results with op
                return op(sf(*self_coords, **self.parameters), of(*other_coords, **other.parameters))

            # Return a new Profile with combined function, axes, and inherited units
            combined_units = op(self.units, other.units) if validate else self.units  # Define unit handling here
            return Profile(combined_function, axes, units=combined_units, **combined_parameters)

        # If `other` is a scalar, apply operation directly to `self._function`
        else:
            def scalar_function(*coords):
                return op(self._function(*coords, **self.parameters), other)

            return Profile(scalar_function, self.axes, units=self.units, **self.parameters)

    def __add__(self, other):
        return self._do_operation(other, op=lambda x, y: x + y, validate=True)

    def __radd__(self, other: float) -> "Profile":
        return self.__add__(other)

    def __mul__(self, other):
        return self._do_operation(other, op=lambda x, y: x * y, validate=False)

    def __sub__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x - y, validate=True)

    def __rsub__(self, other: float) -> "Profile":
        return self._do_operation(other, lambda x, y: y - x, validate=True)

    def __rmul__(self, other: float) -> "Profile":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Profile", float]) -> "Profile":
        return self._do_operation(other, lambda x, y: x / y)

    def __rtruediv__(self, other: float) -> "Profile":
        return self._do_operation(other, lambda x, y: y / x)

    def to_hdf5(self, h5_obj: Union[h5py.File, h5py.Group], group_name: str, overwrite: bool = False):
        """
        Save the Profile instance to a specified group in an HDF5 object.

        Parameters
        ----------
        h5_obj : h5py.File or h5py.Group
            The HDF5 object where the Profile will be saved.
        group_name : str
            The name of the group to store Profile data.
        overwrite : bool, optional
            If True, overwrites existing group with the same name.
        """
        # Check if the group already exists and handle overwriting
        if group_name in h5_obj:
            if overwrite:
                del h5_obj[group_name]  # Delete existing group if overwrite is enabled
            else:
                raise ValueError(f"Group '{group_name}' already exists in the HDF5 file. Use overwrite=True to replace it.")

        # Create the new group for saving the profile data
        group = h5_obj.create_group(group_name)

        # Store class name, axes, and units as attributes
        group.attrs["class_name"] = self.__class__.__name__
        group.attrs["axes"] = np.array(self.axes, dtype='S')  # Store as bytes
        group.attrs["units"] = str(self.units)  # Store units as a string

        # Store parameters in the group's attributes
        for key, value in self.parameters.items():
            group.attrs[key] = value

        # Serialize the function with dill and save it as a dataset
        func_data = dill.dumps(self._function)
        group.create_dataset("function", data=np.void(func_data))  # Store as binary blob

    @classmethod
    def from_hdf5(cls, h5_obj: Union[h5py.File, h5py.Group], group_name: str) -> "Profile":
        """
        Load a Profile instance from a specified group in an HDF5 object.

        Parameters
        ----------
        h5_obj : h5py.File or h5py.Group
            The HDF5 object from which the Profile will be loaded.
        group_name : str
            The name of the group to load Profile data from.

        Returns
        -------
        Profile
            The loaded Profile instance.
        """
        # Check if the group exists
        if group_name not in h5_obj:
            raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")

        # Access the group containing the profile data
        group = h5_obj[group_name]

        # Load the class name and find the correct class in the registry
        class_name = group.attrs["class_name"]

        if class_name in cls.REGISTRY.registry:
            return cls.REGISTRY.registry.get(class_name).from_hdf5(h5_obj,group_name)
        elif class_name == 'Profile':
            pass
        else:
            raise ValueError(f"Class '{class_name}' not found in the registry.")

        # Load axes and units from attributes
        axes = list(group.attrs["axes"].astype(str))
        units = group.attrs["units"]

        # Load parameters from group attributes
        parameters = {key: group.attrs[key] for key in group.attrs if key not in ["class_name", "axes", "units"]}

        # Load and deserialize the function
        func_data = group["function"][()]
        function = dill.loads(bytes(func_data))

        # Return a new instance of the correct Profile subclass with loaded data
        return cls(function, axes, units, **parameters)

    def partial_derivative(self, x: np.ndarray, axes: List[str]|str, **kwargs) -> np.ndarray:
        """
        Compute the partial derivative with respect to a specific axis at a given position.

        Parameters
        ----------
        axes : List[str] or str
            The axis with respect to which the derivative is taken (must be one of self.axes).
        x : np.ndarray
            The coordinates at which to evaluate the derivative. This should be a ``(N,NDIM)`` array
            where ``N`` is the number of input sets to pass through and ``NDIM`` is the number of inputs per
            input set.
        kwargs:
            Additional kwargs to pass to :py:func:`utilities.math.function_partial_derivative`.

        Returns
        -------
        float
            The partial derivative of the profile with respect to the specified axis at the given coordinates.

        Raises
        ------
        ValueError
            If the specified axis is not in the profile's axes.
        """
        # Determine the conversion from the axes (strings) to the relevant integers.
        axes = [self.axes.index(axis) for axis in axes]

        return function_partial_derivative(self,x,axes,**kwargs)


class FixedProfile(Profile):
    AXES: List[str] = None
    UNITS: Union[str,unyt.Unit] = None
    PARAMETERS: Dict[str,Any] = None
    FUNCTION: Callable = None

    def __init__(self,**kwargs):
        # Load the parameters using the defaults and those provided
        # in kwargs.
        kwargs = self._validate_kwargs(kwargs)

        # Perform the superclass load
        super().__init__(self.__class__.FUNCTION,self.__class__.AXES,units=self.__class__.UNITS,**kwargs)

    def _validate_kwargs(self,kwargs):
        _new_kwargs = self.__class__.PARAMETERS.copy()

        for k,v in kwargs.items():
            if k in _new_kwargs:
                _new_kwargs[k] = v
            else:
                raise ValueError

            if _new_kwargs[k] is None:
                raise ValueError

        return _new_kwargs

    @classmethod
    def from_hdf5(cls, h5_obj: Union[h5py.File, h5py.Group], group_name: str) -> "FixedProfile":
        # Check if the group exists
        if group_name not in h5_obj:
            raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")

        # Access the group containing the profile data
        group = h5_obj[group_name]


        parameters = {key: group.attrs[key] for key in group.attrs if key not in ["class_name", "axes", "units", "is_fixed"]}
        return cls( **parameters)
