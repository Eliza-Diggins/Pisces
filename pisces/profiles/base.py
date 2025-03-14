"""
Profile base classes.

Profiles are one of the core components of Pisces, enabling users to model and manipulate mathematical profiles
with symbolic and numerical capabilities. Profiles can represent various physical and mathematical entities, such as
density, mass, or temperature distributions.

For readers who are not familiar with the usage of profiles, we suggest reading :ref:`profiles_overview`. For developers,
the :ref:`profiles_developers` is worth a read before diving into the API.

"""
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

import h5py
import numpy as np
import sympy as sp
from unyt import Unit, unyt_array, unyt_quantity

from pisces.utilities.general import find_in_subclasses
from pisces.utilities.logging import devlog

if TYPE_CHECKING:
    pass


def class_expression(name: str = None, on_demand: bool = True):
    """
    Decorator to register a symbolic expression at the class level.

    This decorator is designed to streamline the definition of class-level symbolic expressions for subclasses of
    :py:class:`Profile`. These expressions are derived from the symbolic representation of the profile's function
    and are shared across all instances of the class.

    Parameters
    ----------
    name : str, optional
        The name of the derived symbolic expression. If not provided, the name of the decorated method will be used.
        This is the name by which the expression is registered and then looked up by the user. It therefore should
        generally be intuitive.
    on_demand : bool, optional
        If ``True`` (default), this won't be loaded until it is requested. Otherwise it is loaded as soon as the
        class is instantiated.
    Returns
    -------
    Callable
        The decorated method.

    Notes
    -----
    To use this decorator, define a method in your :py:class:`Profile` subclass that computes the desired symbolic
    expression. The decorated method must have the following signature:

    .. code-block:: python

        class ProfileClass(Profile):

            @class_expression(name='my_expression')
            def method(
                axes: List[sp.Symbol],
                parameters: Dict[str, sp.Symbol],
                expression: sp.Basic): -> sp.Basic

                # Insert the code to manipulate the axes, parameters, and expression
                # as needed here.

                return expression

    Here, the arguments are as follows:

    - **axes**: A list of ``sympy.Symbol`` objects representing the symbolic axes of the profile.
    - **parameters**: A dictionary mapping parameter names to their ``sympy.Symbol`` representations.
    - **expression**: The symbolic function of the profile.

    The decorated method should return a symbolic expression derived from these inputs. The returned expression is
    automatically registered at the class level under the specified or inferred name.

    Tips
    ----
    - Use this decorator for expressions that are static and depend only on the symbolic representation of the profile,
      not on instance-specific parameters.
    - Derived expressions can be used to add meaningful symbolic attributes to the class, such as analytical derivatives
      or scaling factors.

    """

    def decorator(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            return func(cls, *args, **kwargs)

        wrapper.class_expression = True
        wrapper.expression_name = name or func.__name__
        wrapper.on_demand = on_demand
        return wrapper

    return decorator


class ProfileMeta(ABCMeta):
    """
    Metaclass parent for all profiles in Pisces. This metaclass ensures that all of the
    relevant class attributes are present and specified and that the symbolic attributes are
    correctly manipulated.
    """

    def __new__(mcs, name, bases, cls_dict):
        # Identify if the class is abstract or concrete. This step is necessary so that
        # we don't force abstract classes to have properties which are not necessary given
        # that they are never instantiated.
        is_parent = cls_dict.get("_is_parent_profile", False)

        # Create the base class object without adulteration from the
        # metaclass.
        cls = super().__new__(mcs, name, bases, cls_dict)

        # If the class is not an abstract base class, we need to register
        # symbols and parse the relevant class-level expressions.
        cls._expression_dictionary = {}
        if not is_parent:
            mcs._validate_class(cls)
            mcs._generate_symbolics(cls)
            mcs._register_class_expressions(cls, cls_dict)

        return cls

    @staticmethod
    def _validate_class(cls):
        """
        This method simply checks over the class and ensures that it is self-consistent.
        """
        # Check for the required components of the class.
        _required_components = [
            "AXES",
            "DEFAULT_PARAMETERS",
            "_profile",
            "_expression_dictionary",
            "DEFAULT_AXES_UNITS",
            "SYMBAXES",
            "SYMBPARAMS",
            "profile_expression",
        ]

        for component in _required_components:
            if not hasattr(cls, component):
                raise SyntaxError(
                    "Class '{}' has no attribute '{}'".format(cls.__name__, component)
                )

        # Ensure that the axes units are valid and that the default parameters all have units as well.
        cls.DEFAULT_AXES_UNITS = {
            au: Unit(av) if av is not None else Unit("")
            for au, av in getattr(cls, "AXES_UNITS").items()
        }
        cls.DEFAULT_PARAMETERS = {
            pn: pv if hasattr(pv, "units") else unyt_quantity(pv, "")
            for pn, pv in getattr(cls, "DEFAULT_PARAMETERS").items()
        }

    @staticmethod
    def _generate_symbolics(cls):
        """
        Generate the symbolic attributes for the profile class.
        """
        # Fetch the axes and the parameters and the profile.
        AXES, DEFAULT_PARAMETERS = getattr(cls, "AXES_UNITS"), getattr(
            cls, "DEFAULT_PARAMETERS"
        )
        _profile = getattr(cls, "_profile")

        # Build the sympy symbols for each of the symbolic axes
        # and for each of the parameters.
        cls.SYMBAXES = [sp.Symbol(ax) for ax in AXES]
        cls.SYMBPARAMS = {param: sp.Symbol(param) for param in DEFAULT_PARAMETERS}

        # Attempt to construct the class level function by sending the symbolic attributes
        # through the function to generate an expression. If this fails, we get an error.
        try:
            cls.profile_expression = _profile(*cls.SYMBAXES, **cls.SYMBPARAMS)
        except Exception as e:
            raise ValueError(
                f"Failed to generate symbolic function for '{cls.__name__}' due to: {e}"
            )

    @staticmethod
    def _register_class_expressions(cls, clsdict):
        """
        Register class-level symbolic expressions from decorated methods.

        This method identifies methods in the class dictionary (`clsdict`) that are decorated with
        the `@class_expression` decorator. For each such method, it:

        1. Retrieves the symbolic expression generated by the method using the class's symbolic axes,
           symbolic parameters, and main symbolic function expression (`_func_expr`).
        2. Registers the resulting expression under the name specified in the decorator (or the method name
           if not explicitly specified) using the `set_class_expression` method.

        Parameters
        ----------
        cls : Type['Profile']
            The Profile subclass for which class-level expressions are being registered.
        clsdict : dict
            The dictionary of attributes and methods belonging to the Profile subclass.

        Raises
        ------
        ValueError
            If a decorated method fails to generate a valid symbolic expression.

        Notes
        -----
        - This method assumes that decorated methods accept three arguments:
          `SYMBAXES` (list of symbolic axes), `SYMBPARAMS` (dict of symbolic parameters),
          and `func_expr` (the symbolic function expression of the profile).
        - The expressions registered via this method are shared across all instances of the Profile class.
        """
        seen = set()
        for base in cls.__mro__:
            # You might skip object or other base classes if desired
            if base is object:
                continue
            for attr_name, method in base.__dict__.items():
                if (base, attr_name) in seen:
                    continue
                seen.add((base, attr_name))

                if callable(method) and getattr(method, "class_expression", False):
                    expression_name = getattr(
                        method, "expression_name"
                    )  # Retrieve the name for the expression
                    on_demand = getattr(method, "on_demand", True)
                    devlog.debug(
                        "Registering class expression %s to class %s. (ON_DEMAND=%s)",
                        expression_name,
                        cls.__name__,
                        on_demand,
                    )

                    if not on_demand:
                        devlog.debug(
                            "Evaluating class expression %s...", expression_name
                        )
                        try:
                            # Generate the symbolic expression using the method
                            expression = method(
                                cls.SYMBAXES, cls.SYMBPARAMS, cls.profile_expression
                            )
                            # Register the expression at the class level
                            cls._expression_dictionary[expression_name] = expression
                        except Exception as e:
                            raise ValueError(
                                f"Failed to register class-level expression '{expression_name}' for "
                                f"class '{cls.__name__}' due to: {e}"
                            )
                    else:
                        cls._expression_dictionary[expression_name] = attr_name


class Profile(ABC, metaclass=ProfileMeta):
    """
    Core base class for Pisces profiles.
    """

    # @@ FLAGS @@ #
    # These flags can be used to direct the metaclass on how to handle the class.
    _is_parent_profile: bool = True  # Skip symbolic registration (useful for a 'parent profile' that can't actually be used).

    # @@ AXES AND PARAMETER SETUP @@ #
    # These class attributes specify the base structure of the profile including
    # what independent axes are available and what parameters exist (and what units are associated with
    # each axis). Subclasses need to alter these to suit their use case.
    AXES: List[str] = None
    """ list of str: The axes (free parameters) of the profile.
        These should be strings and be specified in order. The metaclass will process
        them into the symbolic axes of :py:attr:`Profile.SYMBAXES`.
    """
    DEFAULT_PARAMETERS: Dict[str, Any] = None
    """ dict of str, float or :py:class:`unyt.unyt_quantity`: Default parameters for this profile class.
    This dictionary contains the default parameters for the profile with each key representing a parameter name and
    each value corresponding to the default value.

    The units of the values in :py:attr:`DEFAULT_PARAMETERS` are interpreted as the default parameter units. If a float
    is specified, then it is assumed that the parameter is unitless. The units for particular parameters can be changed
    at ``__init__``.
    """
    DEFAULT_AXES_UNITS: Dict[str, Unit] = None
    """ dict of str, :py:class:`unyt.Unit`: The units to associate with each independent variable.
    Each unit should be specified as an unyt unit object or ``None``. If ``None`` is used, then we assume that this
    axis is just a float.
    """

    # @@ SYMBOL CONTAINERS @@ #
    # These class attributes hold the various class-level symbolic objects that are needed for
    # profile manipulation.
    # These should NOT BE SET, they are set dynamically by the metaclass.
    _expression_dictionary: dict[str, sp.Basic] = None
    """ dict[str, sp.Basic]: The dictionary of symbolic expressions for this class."""
    profile_expression: sp.Basic = None
    """ sp.Basic: Symbolic representation of this profile."""
    SYMBAXES: List[sp.Symbol] = None
    """ list of sp.Symbol: The symbolic axes of the profile."""
    SYMBPARAMS: Dict[str, sp.Symbol] = None
    """ dict of sp.Symbol, float: The symbolic parameters of the profile."""

    # @@ INITIALIZATION PROCEDURES @@ #
    # All of these methods play a role in the initialization process.
    def _setup_parameters(
        self, parameter_values: Dict[str, Union[unyt_quantity, float]]
    ) -> Dict[str, unyt_quantity]:
        """
        Set up the parameters for the profile. This function processes the parameter values,
        axis and parameter units, and determines the correct (permanent) units for the profile.

        Parameters
        ----------
        parameter_values : dict
            Dictionary of parameter values.

        Returns
        -------
        dict of str, unyt_quantity
            Dictionary of parameter values with correct units.

        Raises
        ------
        ValueError
            If a provided parameter cannot be converted to the default units.
        """
        # Copy default parameters to avoid modifying class attributes
        _params = self.DEFAULT_PARAMETERS.copy()

        for param, value in parameter_values.items():
            if param not in _params:
                raise KeyError(f"Invalid parameter '{param}' provided.")

            # Determine the unit, defaulting to the parameter's default unit
            units = getattr(value, "units", _params[param].units)

            # Validate unit conversion
            try:
                units.get_conversion_factor(_params[param].units)
            except Exception:
                raise ValueError(
                    f"Cannot convert units for parameter '{param}'. Expected compatible units with {_params[param].units}."
                )

            # Assign the value with the determined units
            _params[param] = unyt_quantity(value, units)

        return _params

    def _setup_axes_units(
        self, axes_units: Dict[str, Union[str, Unit]]
    ) -> Dict[str, Unit]:
        """
        Validate and set up axis units for the profile.

        This function ensures that provided axis units are compatible with the default
        units specified in `AXES_UNITS`, filling in defaults where needed.

        Parameters
        ----------
        axes_units : dict
            Dictionary mapping axis names to their specified units.

        Returns
        -------
        dict of str, Unit
            A dictionary mapping each axis to its corresponding unit.

        Raises
        ------
        KeyError
            If an axis in `axes_units` is not defined in `AXES_UNITS`.
        ValueError
            If a provided axis unit is incompatible with the expected default unit.
        """
        # Copy default axis units to avoid modifying class attributes
        axis_units = self.DEFAULT_AXES_UNITS.copy()
        axes_units = axes_units or {}

        for axis, unit in axes_units.items():
            if axis not in axis_units:
                raise KeyError(
                    f"Invalid axis '{axis}' provided. Expected one of {list(axis_units.keys())}."
                )

            # Determine the unit, defaulting to the axis' default unit
            assigned_unit = getattr(unit, "units", axis_units[axis])

            # Validate unit conversion
            try:
                assigned_unit.get_conversion_factor(axis_units[axis])
            except Exception:
                raise ValueError(
                    f"Incompatible units for axis '{axis}'. Expected compatible units with {axis_units[axis]}."
                )

            # Assign the validated unit
            axis_units[axis] = assigned_unit

        return axis_units

    def _set_output_units(self):
        return self._profile(
            *[1 * au for au in self.axes_units.values()], **self._parameters
        ).units

    def _validate_units(self):
        # This function can be used to check that units are valid if necessary.
        pass

    def _setup_call_function(self):
        """
        Set up the callable function by substituting parameter values
        and creating a numerical function.
        """
        # Symbolically substitute in the parameters for this profile.
        symbolic_expr = self.substitute_expression(self.__class__.profile_expression)

        # Set up the call function (symbolically) and then
        # set the call function numerical via lambdify.
        self._call_function_symbolic = symbolic_expr
        self._call_function = self.lambdify_expression(symbolic_expr)

    def __init__(self, axes_units: Dict[str, Union[str, Unit]] = None, **kwargs):
        # Configure the instances parameters and the corresponding units.
        self._parameters: Dict[str, unyt_quantity] = self._setup_parameters(kwargs)
        self._axes_units: Dict[str, Unit] = self._setup_axes_units(axes_units)

        # Set up the call function. This simply takes the class level
        # expression and substitutes the values of the various parameters into
        # it.
        self._setup_call_function()

        # Set up the output units.
        self._output_units: Unit = self._set_output_units()
        self._validate_units()

        # Set up the repository for derived symbolics and lambdified methods
        self._inst_expression_dictionary: Dict[str, sp.Basic] = {}
        self._inst_numeric_dictionary: Dict[str, Callable] = {}

    # @@ PROPERTIES AND BASIC MUTABILITY @@ #
    @property
    def parameters(self) -> Dict[str, unyt_quantity]:
        """
        Provides an immutable copy of the parameters for this profile.
        """
        return self._parameters.copy()

    @property
    def axes_units(self) -> Dict[str, Unit]:
        """
        Provides an immutable copy of the units for each of the independent axes.
        """
        return self._axes_units.copy()

    @property
    def symbolic_expression(self) -> sp.Basic:
        """
        The symbolic (sympy) expression for this profile with all the parameters
        substituted into the expression.
        """
        return self._call_function_symbolic

    @property
    def class_symbolic_expression(self) -> sp.Basic:
        """
        The symbolic (sympy) expression for this profile with all the parameters retained
        as symbols.
        """
        return self.__class__.profile_expression

    @property
    def output_units(self) -> Unit:
        """
        The natural units in which the results of this profile are expressed.
        """
        return self._output_units

    # @@ Class Expressions @@ #
    # These methods manage interactions with the class-level expressions.
    @classmethod
    def get_class_expression(cls, expression_name: str) -> sp.Basic:
        """
        Retrieve a symbolic expression derived from this profile without any parameter substitutions.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve.

        Returns
        -------
        sp.Basic
            The requested symbolic expression.

        Raises
        ------
        KeyError
            If the requested expression name does not exist.
        """
        # Check if the expression name is in the expression dictionary. If it is,
        # we have the expression and just need to perform processing if It's needed before
        # returning.
        if expression_name in cls._expression_dictionary:
            _class_expr = cls._expression_dictionary[expression_name]

            # Check if the expression is a string. If it is, we have on-demand loading of the
            # symbolic profile so we need to generate a symbolic equivalent.
            if isinstance(_class_expr, str):
                devlog.debug("Evaluating class expression %s...", expression_name)
                try:
                    _class_expr = getattr(cls, _class_expr)(
                        cls.SYMBAXES, cls.SYMBPARAMS, cls.profile_expression
                    )
                    cls._expression_dictionary[expression_name] = _class_expr
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to register expression '{expression_name}' at class level. ERROR: {e}"
                    ) from e

            return _class_expr

        else:
            raise KeyError(f"Class-level expression '{expression_name}' not found.")

    @classmethod
    def list_class_expressions(cls) -> List[str]:
        """
        List the available class-level expressions.

        Returns
        -------
        list of str
            The list of available class-level expressions.
        """
        return list(cls._expression_dictionary.keys())

    @classmethod
    def has_class_expression(cls, expression_name: str) -> bool:
        """
        Check if a symbolic expression is registered at the class level.

        Parameters
        ----------
        expression_name: str
            The name of the symbolic expression to check.

        Returns
        -------
        bool
            ``True`` if the symbolic expression is registered at the class level.
        """
        return expression_name in cls._expression_dictionary

    # @@ Instance Expressions @@ #
    # These methods manage the instance level expressions for the
    # profiles.
    def get_expression(self, expression_name: str) -> sp.Basic:
        """
        Retrieve a symbolic expression derived from this profile given the name of the
        derived quantity. The returned profile already has parameter substitutions.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve.

        Returns
        -------
        sp.Basic
            The requested symbolic expression.

        Raises
        ------
        KeyError
            If the expression is not found at either the instance or class level.
        """
        # Look for the expression in the instance directory first.
        if expression_name in self._inst_expression_dictionary:
            return self._inst_expression_dictionary[expression_name]

        # We couldn't find it in the instance directory, now we try to fetch it
        # and perform a substitution.
        if expression_name in self.__class__._expression_dictionary:
            self._inst_expression_dictionary[
                expression_name
            ] = self.substitute_expression(
                self.__class__._expression_dictionary[expression_name]
            )
            return self._inst_expression_dictionary[expression_name]

        raise KeyError(f"Expression '{expression_name}' not found.")

    def set_expression(
        self, expression_name: str, expression: sp.Basic, overwrite: bool = False
    ):
        """
        Set a symbolic expression at the instance level.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to register.
        expression : sp.Basic
            The symbolic expression to register.
        overwrite : bool, optional
            If True, overwrite an existing expression with the same name. Defaults to False.

        Raises
        ------
        ValueError
            If the expression name already exists and `overwrite` is False.
        """
        if (expression_name in self._inst_expression_dictionary) and (not overwrite):
            raise ValueError(
                f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it."
            )
        self._inst_expression_dictionary[expression_name] = expression

    def list_expressions(self, include_class_level: bool = True) -> List[str]:
        """
        List the available instance-level expressions.

        Parameters
        ----------
        include_class_level : bool, optional
            If True, include the class level symbolic expressions. Defaults to True.

        Returns
        -------
        list of str
            The list of available class-level expressions.
        """
        return list(
            set(self._expression_dictionary.keys())
            | set(self._inst_expression_dictionary.keys())
        )

    def has_expression(self, expression_name: str) -> bool:
        """
        Check if a symbolic expression is registered at the instance level.

        Parameters
        ----------
        expression_name: str
            The name of the symbolic expression to check.

        Returns
        -------
        bool
            ``True`` if the symbolic expression is registered.
        """
        return expression_name in self._inst_expression_dictionary

    def get_numeric_expression(self, expression_name: str) -> Callable:
        """
        Retrieve or create a numeric (callable) version of a symbolic expression.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve or convert.

        Returns
        -------
        Callable
            A numeric (callable) version of the symbolic expression.

        Raises
        ------
        KeyError
            If the symbolic expression is not found.
        """
        if expression_name not in self._inst_numeric_dictionary:
            symbolic_expression = self.get_expression(expression_name)
            self._inst_numeric_dictionary[expression_name] = self.lambdify_expression(
                symbolic_expression
            )
        return self._inst_numeric_dictionary[expression_name]

    # @@ Utility Functions @@ #
    # These utility methods provide interaction for the symbolic / numerical
    # expressions and some other features of the class which are useful.
    def substitute_expression(self, expression: Union[str, sp.Basic]) -> sp.Basic:
        """
        Substitute the parameter values (:py:attr:`Profile.parameters`) into a symbolic expression.

        Parameters
        ----------
        expression : str or sp.Basic
            The symbolic expression to substitute into.

        Returns
        -------
        sp.Basic
            Substituted symbolic expression.
        """
        # Substitute in each of the parameter values.
        _params = {k: v.d for k, v in self._parameters.items()}
        return sp.sympify(expression).subs(_params)

    def lambdify_expression(self, expression: Union[str, sp.Basic]) -> Callable:
        """
        Convert a symbolic expression into a callable function.

        Parameters
        ----------
        expression : str or sp.Basic
            The symbolic expression to lambdify.

        Returns
        -------
        Callable
            A callable numerical function.
        """
        expression = sp.sympify(expression)
        return sp.lambdify(self.__class__.SYMBAXES, expression, "numpy")

    @staticmethod
    @abstractmethod
    def _profile(*args, **kwargs):
        pass

    def __call__(self, *args, units: Union[str, Unit] = None) -> Any:
        """
        Evaluate the profile's mathematical function with the given inputs.

        This method allows the profile instance to be used as a callable, evaluating
        its associated function with the provided input values for the independent variables.

        Parameters
        ----------
        *args : tuple
            Input values corresponding to the independent variables defined in the :py:attr:`Profile.AXES` class attribute.
            The number and order of inputs must match the defined axes.
        units: str or Unit, optional
            The output units to use.

        Returns
        -------
        Any
            The computed result of the profile's function evaluated at the specified inputs.

        Raises
        ------
        ValueError
            If the number of input arguments does not match the number of independent variables (:py:attr:`Profile.AXES`).
        TypeError
            If the input values are incompatible with the profile's function.

        Notes
        -----
        - The profile's function is generated during initialization using symbolic expressions and lambdified for numerical evaluation.
        - The inputs provided must conform to the expected type and dimensionality required by the function.
        - This method provides a convenient interface for directly evaluating the profile, e.g., `result = profile(x, y, z)`.

        """
        # Validate that the arguments have the correct length.
        if len(args) != len(self.AXES):
            raise ValueError(f"Expected {len(self.AXES)} arguments, got {len(args)}.")

        # Coerce the arg units.
        args = [
            arg.to_value(unit) if hasattr(arg, "units") else arg
            for arg, unit in zip(args, self.axes_units.values())
        ]

        # Coerce the output units.
        if units is None:
            units = self.output_units

        return unyt_array(self._call_function(*args), self.output_units).to(units)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"<{self.__class__.__name__}({param_str})>"

    def __str__(self) -> str:
        return self.__repr__()

    # @@ IO PROCEDURES @@ #
    # These methods manage reading and writing profiles to HDF5.
    def to_hdf5(
        self,
        h5_obj: Union[h5py.File, h5py.Group],
        group_name: str,
        overwrite: bool = False,
    ):
        """
        Save this profile to an HDF5 file. The HDF5 should be opened using ``h5py`` and
        provided as ``h5_obj``. This method saves the data for the profile into a specific group (``group_name``),
        which will be created if necessary.

        ``overwrite`` will determine whether an existing group is overwritten.

        Parameters
        ----------
        h5_obj : h5py.File or h5py.Group
            HDF5 object where the Profile will be saved.
        group_name : str
            Name of the group in the HDF5 file.
        overwrite : bool, optional
            If True, overwrite existing group with the same name.

        Raises
        ------
        ValueError
            If the group already exists and overwrite is False.
        """
        # Validate overwrite scenarios.
        if (group_name in h5_obj) and overwrite:
            # Remove the group and prepare to overwrite.
            del h5_obj[group_name]
        elif (group_name in h5_obj) and not overwrite:
            raise ValueError(
                f"Group '{group_name}' already exists. Use `overwrite=True` to replace it."
            )

        # Create the HDF5 group object into which the data should get written.
        group = h5_obj.create_group(group_name)

        # Save the data to HDF5. The basic scheme here is to write out the class name
        # to that a backward search can identify the correct class on read and then to
        # provide the axes units and parameters as attributes.
        group.attrs["class_name"] = self.__class__.__name__
        group.attrs["axes_units"] = np.asarray(self.axes_units.values(), dtype="S")

        # Write out each of the parameter values and add a units version as well.
        for key, value in self.parameters.items():
            group.attrs[key] = value
            group.attrs[key + "_u"] = str(value.units)

    @classmethod
    def from_hdf5(
        cls, h5_obj: Union[h5py.File, h5py.Group], group_name: str
    ) -> "Profile":
        """
        Load a :py:class:`Profile` instance from disk. The HDF5 file should have a specific
        group (``group_name``), which contains the data for the profile.

        Parameters
        ----------
        h5_obj : h5py.File or h5py.Group
            HDF5 object from which the Profile will be loaded.
        group_name : str
            Name of the group in the HDF5 file.

        Returns
        -------
        Profile
            An instance of the loaded Profile subclass.

        Raises
        ------
        ValueError
            If the group does not exist in the HDF5 file or required attributes are missing.
        """
        # Check that the group actually exists and then access it
        # so that we can directly interact with the data.
        if group_name not in h5_obj:
            raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")
        group = h5_obj[group_name]

        # Find the correct subclass from which to load this profile. This will
        # search all the descendant classes and find any matches.
        class_name = group.attrs.get("class_name", None)
        _subcls = find_in_subclasses(cls, class_name)

        # Read the units for the axes - these are the only "special" parameters that
        # need to be handled.
        # noinspection PyUnresolvedReferences
        axes_units = group.attrs.get("axes_units", list(_subcls.AXES_UNITS.values()))
        # noinspection PyUnresolvedReferences
        axes_units = {k: axes_units[i] for i, k in enumerate(_subcls.AXES_UNITS.keys())}

        # Now parse through all the core parameters and their units.
        _parameter_names = [
            key
            for key in group.attrs
            if (key not in ["class_name", "axes_units"]) and (not key.endswith("_u"))
        ]

        parameters = {
            _pn: unyt_quantity(group.attrs[_pn], group.attrs.get(_pn + "_u", ""))
            for _pn in _parameter_names
        }
        # noinspection PyCallingNonCallable
        return _subcls(axes_units=group.attrs["units"], **parameters)


class RadialProfile(Profile, ABC):
    r"""
    Abstract base class for radial profiles.

    This class extends the :py:class:`Profile` to include methods specific to radial profiles,
    particularly the ability to extract limiting power-law behaviors at small (inner) or large (outer)
    radii.
    """
    _is_parent_profile = True

    # Class attribute for the radial axis
    AXES: List[str] = ["r"]
    """
    list of str: The axes (free parameters) of the profile.
        These should be strings and specified in order. The metaclass processes
        them into symbolic axes via :py:attr:`Profile.SYMBAXES`.
    """
    DEFAULT_AXES_UNITS = {"r": Unit("pc")}

    @class_expression("derivative", on_demand=True)
    @staticmethod
    def _r_derivative(axes, params, expression):
        # Determines the derivative of the profile with respect to radius.
        # This is ON_DEMAND, so we only grab this when it's requested.
        return sp.simplify(sp.diff(expression, axes[0]))


class CylindricalProfile(Profile, ABC):
    r"""
    Abstract base class for cylindrical profiles.
    """
    _is_parent_profile = True

    # Class attribute for the radial axis
    AXES: List[str] = ["r", "z"]
    """
    list of str: The axes (free parameters) of the profile.
        These should be strings and specified in order. The metaclass processes
        them into symbolic axes via :py:attr:`Profile.SYMBAXES`.
    """
    DEFAULT_AXES_UNITS = {"r": Unit("pc"), "z": Unit("pc")}
