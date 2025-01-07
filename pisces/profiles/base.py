"""
Profile base classes.

Profiles are one of the core components of Pisces, enabling users to model and manipulate mathematical profiles
with symbolic and numerical capabilities. Profiles can represent various physical and mathematical entities, such as
density, mass, or temperature distributions.

For readers who are not familiar with the usage of profiles, we suggest reading :ref:`profiles_overview`. For developers,
the :ref:`profiles_developers` is worth a read before diving into the API.

"""
from abc import ABC, ABCMeta
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import h5py
import numpy as np
from unyt import Unit

from pisces.utilities.general import find_in_subclasses

if TYPE_CHECKING:
    pass

from functools import wraps
from typing import Any, Dict, Tuple, Type

import sympy as sp

from pisces.utilities.logging import devlog


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


# noinspection PyTypeChecker
class ProfileMeta(ABCMeta):
    """
    A metaclass for managing Profile subclasses with automatic symbolic attribute generation.

    This metaclass handles the following tasks:

    1. Generates symbolic representations (``SYMBAXES``, ``SYMBPARAMS``, ``_func_expr``) for axes,
       parameters, and the class-level ``_function``.
    2. Derive additional symbolic expressions for the class.

    Abstract classes are ignored for both registration and symbolic generation.
    """

    def __new__(
        mcs: Type["ProfileMeta"],
        name: str,
        bases: Tuple[Type, ...],
        clsdict: Dict[str, Any],
    ) -> Type:
        """
        Create a new class, handle registration, and generate symbolic attributes.

        Parameters
        ----------
        mcs : Type[ProfileMeta]
            The metaclass itself.
        name : str
            The name of the class being created.
        bases : tuple
            The base classes of the class being created.
        clsdict : dict
            Dictionary containing the attributes and methods of the class being created.

        Returns
        -------
        Type
            The newly created class.
        """
        # Identify if the class is abstract or concrete. This step is necessary so that
        # we don't force abstract classes to have properties which are not necessary given
        # that they are never instantiated.
        is_abstract = clsdict.get("_IS_ABC", False)

        # Create the base class object without adulteration from the
        # metaclass.
        cls = super().__new__(mcs, name, bases, clsdict)

        # If the class is not an abstract base class, we need to register
        # symbols and parse the relevant class-level expressions.
        cls._EXPR_DICT = {}
        if not is_abstract:
            mcs._generate_symbolics(cls)
            mcs._register_class_expressions(cls, clsdict)

        return cls

    @staticmethod
    def _generate_symbolics(cls: Type["Profile"]):
        """
        Generate symbolic representations for axes, parameters, and the function expression of a Profile class.

        This method processes the `AXES`, `DEFAULT_PARAMETERS`, and `_function` attributes of a Profile subclass to create
        symbolic representations used throughout the class. Specifically, it:

        1. Converts the string representations in `AXES` to `sympy.Symbol` `objects, stored in `SYMBAXES`.
        2. Converts the dictionary keys in `DEFAULT_PARAMETERS` to `sympy.Symbol` objects, stored in `SYMBPARAMS`.
        3. Evaluates the `_function` callable using the symbolic axes and parameters to produce a symbolic
           representation of the function, stored in `_func_expr`.

        Parameters
        ----------
        cls : Type['Profile']
            The Profile subclass for which symbolic representations are generated.

        Raises
        ------
        ValueError
            If any of the following conditions are met:

            - The class is missing the `AXES` attribute.
            - The class is missing the `DEFAULT_PARAMETERS` attribute.
            - The class is missing a callable `_function` attribute.
            - The symbolic function generation fails (e.g., due to invalid inputs or function definition).

        Notes
        -----
        - This method assumes that the `AXES` attribute is a list of strings and `DEFAULT_PARAMETERS` is a dictionary with string keys.
        - `_function` must be a callable that accepts the symbolic axes and parameters as arguments.
        """
        # Validate that we have the necessary basic attributes.
        AXES = getattr(cls, "AXES", None)
        DEFAULT_PARAMETERS = getattr(cls, "DEFAULT_PARAMETERS", None)
        _func = getattr(cls, "_function", None)

        if not AXES:
            raise ValueError(f"Class attribute `AXES` is missing in '{cls.__name__}'.")
        if not DEFAULT_PARAMETERS:
            raise ValueError(
                f"Class attribute `DEFAULT_PARAMETERS` is missing in '{cls.__name__}'."
            )
        if not callable(_func):
            raise ValueError(
                f"Class attribute `_function` is missing or not callable in '{cls.__name__}'."
            )

        # Convert AXES to symbolic representation
        cls.SYMBAXES = [sp.Symbol(ax) for ax in AXES]

        # Convert DEFAULT_PARAMETERS to symbolic representation
        cls.SYMBPARAMS = {param: sp.Symbol(param) for param in DEFAULT_PARAMETERS}

        # Generate the symbolic function expression
        try:
            cls._func_expr = _func(*cls.SYMBAXES, **cls.SYMBPARAMS)
        except Exception as e:
            raise ValueError(
                f"Failed to generate symbolic function for '{cls.__name__}' due to: {e}"
            )

    @staticmethod
    def _register_class_expressions(cls: Type["Profile"], clsdict: Dict[str, Any]):
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
                                cls.SYMBAXES, cls.SYMBPARAMS, cls._func_expr
                            )
                            # Register the expression at the class level
                            cls._EXPR_DICT[expression_name] = expression
                        except Exception as e:
                            raise ValueError(
                                f"Failed to register class-level expression '{expression_name}' for "
                                f"class '{cls.__name__}' due to: {e}"
                            )
                    else:
                        cls._EXPR_DICT[expression_name] = attr_name


class Profile(ABC, metaclass=ProfileMeta):
    """
    Abstract base class for mathematical profiles with symbolic and numerical support.

    This class provides a framework for defining mathematical profiles with symbolic
    expressions, parameterized inputs, and units. It supports automatic symbolic representation,
    validation of input parameters, unit handling, and serialization/deserialization.

    .. rubric:: Implementation Details

    Each :py:class:`Profile` class is characterized by only a couple of special class-level attributes:

    - :py:attr:`Profile.AXES`: Provides a list of the axes that can be passed to the profile. These become
      the arguments of the eventual ``callable`` object the user interacts with.
    - :py:attr:`Profile.DEFAULT_PARAMETERS`: A dictionary with parameter names as keys and their default values. This
      eventually becomes the ``kwargs`` that the user passes during instantiation.
    - :py:attr:`Profile.DEFAULT_UNITS`: A string specifying the default units of the profile. These can be overwritten later, but they
      do define the permissible units as user specified units which are incompatible lead to an error.
    - ``_function``: A class method implementing the symbolic mathematical function of the profile.

    With these 4 attributes defined, the :py:class:`Profile` subclass is fully functional.

    .. rubric:: Expressions and Derived Attributes

    The :py:class:`Profile` class supports both class-level and instance-level derived symbolic expressions. These
    derived expressions can represent additional features or properties of the profile and are managed separately from
    the primary symbolic expression.

    .. hint::

        The idea here is that you might want your custom profile to have certain derived attributes (like a mass profile for
        a particular density profile); however, these should often only be derived once -- at the class level.

    **Class-Level Expressions:**

    Class-level expressions are shared across all instances of the profile class. They are defined and accessed
    using the following methods:

    - **Define**: Use the :py:meth:`Profile.set_class_expression` method to register a symbolic expression.
    - **Access**: Use the :py:meth:`Profile.get_class_expression` method to retrieve a registered symbolic expression.

    .. dropdown:: Example:

        .. code-block:: python

            from sympy import symbols

            class MyProfile(Profile):
                AXES = ['x']
                PARAMETERS = {'a': 1, 'b': 2}
                _function = lambda x, a=1, b=2: a*x**2

            x = MyProfile.SYMBAXES[0]
            a = MyProfile.SYMBPARAMETERS['a']
            MyProfile.set_class_expression('slope', 2*a*x )
            slope_symbolic = MyProfile.get_class_expression('slope')  # Returns the symbolic expression for 'a'

    .. tip::

        For developers: Class-level expressions can also be registered automatically using the :py:func:`class_expression` decorator
        on specific methods. See the documentation for the decorator for more details on how this can be used.

    **Instance-Level Expressions:**

    Instance-level expressions are specific to a particular instance and can override or extend the class-level definitions.
    They are managed using the following methods:

    - **Define**: Use the :py:meth:`Profile.set_expression` method to register a symbolic expression.
    - **Access**: Use the :py:meth:`Profile.get_expression` method to retrieve a registered expression.

    .. important::

        The :py:meth:`Profile.get_expression` will first search for an **instance-level expression** matching the specified
        name; however, if it fails to find one (by default) it will the search the **class-level expressions**. If one is found,
        the :py:attr:`Profile.SYMBPARAMS` are substituted for the parameter values of the profile and the new instance-level expression
        is registered.

    For **instance-level** expressions, the :py:meth:`Profile.get_numeric_expression` method can be used to obtain a "lambdified"
    version of a symbolic expression suitable for numerical calculations.

    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+
    | Method                            | Scope        | Purpose                                                                           |
    +===================================+==============+===================================================================================+
    | :py:meth:`set_class_expression`   | Class-Level  | Define a symbolic expression at the class level.                                  |
    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+
    | :py:meth:`get_class_expression`   | Class-Level  | Retrieve a symbolic expression from the class level.                              |
    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+
    | :py:meth:`set_expression`         | Instance     | Define a symbolic expression at the instance level.                               |
    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+
    | :py:meth:`get_expression`         | Instance     | Retrieve a symbolic expression from the instance or, optionally, the class level. |
    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+
    | :py:meth:`get_numeric_expression` | Instance     | Retrieve or create a callable numeric version of an expression.                   |
    +-----------------------------------+--------------+-----------------------------------------------------------------------------------+


    By combining these features, the :py:class:`Profile` class provides a powerful and flexible framework for
    modeling mathematical profiles with extensible symbolic and numeric capabilities.
    """

    # @@ CLASS FLAGS @@ #
    # These flags are used by the metaclass to determine whether or not to
    # treat the class specially.
    _IS_ABC: bool = True  # Mark the class as an abstract class -- the metaclass skips this class if True.

    # @@ CLASS ATTRIBUTES @@ #
    # Core attributes of the profile class.
    #
    # DEVELOPERS: These should all be set when subclassing Profile.
    AXES: List[str] = None
    """ list of str: The axes (free parameters) of the profile.
        These should be strings and be specified in order. The metaclass will process
        them into the symbolic axes of :py:attr:`Profile.SYMBAXES`.
    """
    DEFAULT_PARAMETERS: Dict[str, Any] = None
    """ dict of str, Any: The parameters of the profile.
        The keys of the dictionary should specify the symbol for the kwarg (its name) and the values
        should represent default values (``float``). These do not possess inherent units, instead, they
        are simply to be provided in a unit consistent way.
    """
    DEFAULT_UNITS: str = ""
    """ str: The default units of the profile.
        These can later be overridden by specifying the ``units`` kwarg at instantiation.

        .. hint::

            These units specify the dimensions of the profile output. If you later specify an instance of
            the profile with units that are inconsistent with :py:attr:`Profile.DEFAULT_UNITS`, and error is raised.

    """

    # @@ CLASS SYMBOL REGISTRIES @@ #
    # These should NEVER be altered -- they just hold the symbols and expressions
    # generated by the metaclass.
    # The _EXPR_DICT is simply a class-level repository for class expressions.
    # The _func_expr holds the class-level function expression.
    _EXPR_DICT: dict[str, sp.Basic] = None
    _func_expr: sp.Basic = None
    SYMBAXES: List[sp.Symbol] = None
    """ list of sp.Symbol: The symbolic axes of the profile."""
    SYMBPARAMS: Dict[str, sp.Symbol] = None
    """ dict of sp.Symbol, float: The symbolic parameters of the profile."""

    def __init__(self, units: Union[str, "Unit"] = None, **kwargs):
        """
        Initialize a Profile instance with specified parameter values and units.

        This constructor initializes the profile by setting its parameters, units, and derived symbolic/numeric expressions.
        It validates the provided parameter values and ensures unit compatibility with the profile's default units.

        Parameters
        ----------
        units : str or Unit, optional
            Units for the profile. If not specified, the class-level default units (:py:attr:`Profile.DEFAULT_UNITS`) will be used.
            Units must be compatible with the profile's defined base units.
        kwargs :
            Parameter values for the profile. Each key must correspond to a parameter defined in the :py:attr:`Profile.DEFAULT_PARAMETERS` class attribute.

        Raises
        ------
        ValueError
            If any provided parameter is not recognized, required parameters are missing, or the specified units are
            incompatible with the profile's base units.

        Notes
        -----
        - Parameters provided in `kwargs` override the default values specified in the class-level `PARAMETERS` attribute.
        - The `units` parameter must either be a valid unit string or a `Unit` instance and must match the dimensionality
          of the profile's base units.
        - The `_setup_call_function` method is invoked to create the callable function representing the profile's mathematical expression.
        - Repositories for storing symbolic and numerical derived attributes (`_derived_symbolics` and `_derived_numerics`) are initialized.
        """
        # Configure instance parameters. Ensures that only recognized parameters
        # are specified and propagates the defaults from the class level PARAMETERS.
        self.parameters: Dict[str, float] = self._validate_and_update_params(kwargs)
        """dict of str, float: The parameters of the profile."""

        # Handle units -- Make sure they are valid when converted to the
        # default. This ensures dimensionality constraints.
        self.units = Unit(units or self.DEFAULT_UNITS)
        """ unyt.Unit: The units for the output of this profile."""
        try:
            _ = self.units.get_conversion_factor(Unit(self.__class__.DEFAULT_UNITS))
        except Exception:
            raise ValueError(
                f"Units {self.units} are not consistent with base units for {self.__class__.__name__} ({self.__class__.DEFAULT_UNITS})."
            )

        # Setup the call function.
        self._setup_call_function()

        # Setup the repository for derived symbolics and lambdified methods
        self._derived_symbolics: Dict[str, sp.Basic] = {}
        self._derived_numerics: Dict[str, Callable] = {}

    def _validate_and_update_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and update the parameters with defaults and provided values.

        Parameters
        ----------
        kwargs : dict
            Parameter values passed during initialization.

        Returns
        -------
        Dict[str, Any]
            Updated parameter dictionary.

        Raises
        ------
        ValueError
            If unexpected parameters are provided or required parameters are missing.
        """
        parameters = self.__class__.DEFAULT_PARAMETERS.copy()

        # Update with user-provided parameters
        for key, value in kwargs.items():
            if key in parameters:
                parameters[key] = value
            else:
                raise ValueError(
                    f"Unexpected parameter: '{key}' not recognized in class '{self.__class__.__name__}'."
                )

        # Check for missing required parameters
        missing = [key for key, value in parameters.items() if value is None]
        if missing:
            raise ValueError(
                f"Missing required parameters for class '{self.__class__.__name__}': {missing}"
            )

        return parameters

    def _setup_call_function(self):
        """
        Setup the callable function by substituting parameter values
        and creating a numerical function.
        """
        symbolic_expr = self.substitute_expression(self.__class__._func_expr)
        self._call_function_symbolic = symbolic_expr
        self._call_function = self.lambdify_expression(symbolic_expr)

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
        return sp.sympify(expression).subs(self.parameters)

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

    def __call__(self, *args) -> Any:
        """
        Evaluate the profile's mathematical function with the given inputs.

        This method allows the profile instance to be used as a callable, evaluating
        its associated function with the provided input values for the independent variables.

        Parameters
        ----------
        *args : tuple
            Input values corresponding to the independent variables defined in the :py:attr:`Profile.AXES` class attribute.
            The number and order of inputs must match the defined axes.

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
        return self._call_function(*args)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"<{self.__class__.__name__}({param_str})>"

    def __str__(self) -> str:
        return self.__repr__()

    def to_hdf5(
        self,
        h5_obj: Union[h5py.File, h5py.Group],
        group_name: str,
        overwrite: bool = False,
    ):
        """
        Save the Profile instance to an HDF5 group.

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
        if group_name in h5_obj:
            if overwrite:
                del h5_obj[group_name]
            else:
                raise ValueError(
                    f"Group '{group_name}' already exists. Use `overwrite=True` to replace it."
                )

        group = h5_obj.create_group(group_name)
        group.attrs["class_name"] = self.__class__.__name__
        group.attrs["axes"] = np.array(self.__class__.AXES, dtype="S")
        group.attrs["units"] = str(self.units)

        for key, value in self.parameters.items():
            group.attrs[key] = value

    @classmethod
    def from_hdf5(
        cls, h5_obj: Union[h5py.File, h5py.Group], group_name: str
    ) -> "Profile":
        """
        Load a Profile instance from an HDF5 group.

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
        if group_name not in h5_obj:
            raise ValueError(f"Group '{group_name}' does not exist in the HDF5 file.")

        group = h5_obj[group_name]

        # Load class name and ensure compatibility
        class_name = group.attrs.get("class_name", None)
        _subcls = find_in_subclasses(cls, class_name)

        # Load parameters and validate
        parameters = {
            key: group.attrs[key]
            for key in group.attrs
            if key not in ["class_name", "axes", "units"]
        }
        return _subcls(units=group.attrs["units"], **parameters)

    @property
    def symbolic_expression(self) -> sp.Basic:
        """
        The symbolic expression of the profile function.
        """
        return self._call_function_symbolic

    @property
    def class_symbolic_expression(self) -> sp.Basic:
        """
        The class-level symbolic expression of the profile function.
        """
        return self.__class__._func_expr

    @classmethod
    def set_class_expression(
        cls, expression_name: str, expression: sp.Basic, overwrite: bool = False
    ):
        """
        Set a symbolic expression at the class level.

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
        if expression_name in cls._EXPR_DICT and not overwrite:
            raise ValueError(
                f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it."
            )
        cls._EXPR_DICT[expression_name] = expression

    @classmethod
    def get_class_expression(cls, expression_name: str) -> sp.Basic:
        """
        Retrieve a symbolic expression registered at the class level.

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
        if expression_name in cls._EXPR_DICT:
            _class_expr = cls._EXPR_DICT[expression_name]

            if isinstance(_class_expr, str):
                # This is an on-demand class expression that needs to now be loaded.
                devlog.debug("Evaluating class expression %s...", expression_name)
                try:
                    _class_expr = getattr(cls, _class_expr)(
                        cls.SYMBAXES, cls.SYMBPARAMS, cls._func_expr
                    )
                    cls._EXPR_DICT[expression_name] = _class_expr
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
        return list(cls._EXPR_DICT.keys())

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
        return expression_name in cls._EXPR_DICT

    def get_expression(
        self, expression_name: str, search_class: bool = False
    ) -> sp.Basic:
        """
        Retrieve an instance-level or class-level symbolic expression.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve.
        search_class : bool, optional
            If True, search for the expression at the class level if not found at the instance level. Defaults to False.

        Returns
        -------
        sp.Basic
            The requested symbolic expression.

        Raises
        ------
        KeyError
            If the expression is not found at either the instance or class level.
        """
        if expression_name in self._derived_symbolics:
            return self._derived_symbolics[expression_name]

        if search_class:
            try:
                return self.substitute_expression(
                    self.get_class_expression(expression_name)
                )
            except KeyError:
                pass

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
        if (
            (expression_name in self._derived_symbolics)
            or (expression_name in self._EXPR_DICT)
        ) and not overwrite:
            raise ValueError(
                f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it."
            )
        self._derived_symbolics[expression_name] = expression

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
        return list(set(self._EXPR_DICT.keys()) | set(self._derived_symbolics.keys()))

    def has_expression(
        self, expression_name: str, include_class_level: bool = True
    ) -> bool:
        """
        Check if a symbolic expression is registered at the instance level.

        Parameters
        ----------
        expression_name: str
            The name of the symbolic expression to check.
        include_class_level : bool, optional
            If True, include the class level symbolic expressions. Defaults to True.

        Returns
        -------
        bool
            ``True`` if the symbolic expression is registered.
        """
        if expression_name in self._derived_symbolics:
            return True

        if include_class_level:
            return expression_name in self._EXPR_DICT

        return False

    def get_numeric_expression(
        self, expression_name: str, search_class: bool = False
    ) -> Callable:
        """
        Retrieve or create a numeric (callable) version of a symbolic expression.

        Parameters
        ----------
        expression_name : str
            The name of the symbolic expression to retrieve or convert.
        search_class : bool, optional
            If True, search for the expression at the class level if not found at the instance level. Defaults to False.

        Returns
        -------
        Callable
            A numeric (callable) version of the symbolic expression.

        Raises
        ------
        KeyError
            If the symbolic expression is not found.
        """
        if expression_name not in self._derived_numerics:
            symbolic_expression = self.get_expression(
                expression_name, search_class=search_class
            )
            self._derived_numerics[expression_name] = self.lambdify_expression(
                symbolic_expression
            )
        return self._derived_numerics[expression_name]


class RadialProfile(Profile, ABC):
    r"""
    Abstract base class for radial profiles.

    This class extends the :py:class:`Profile` to include methods specific to radial profiles,
    particularly the ability to extract limiting power-law behaviors at small (inner) or large (outer)
    radii.
    """
    _IS_ABC = True

    # Class attribute for the radial axis
    AXES: List[str] = ["r"]
    """
    list of str: The axes (free parameters) of the profile.
        These should be strings and specified in order. The metaclass processes
        them into symbolic axes via :py:attr:`Profile.SYMBAXES`.
    """

    @class_expression("derivative", on_demand=True)
    @staticmethod
    def _r_derivative(axes, params, expression):
        # Determines the derivative of the profile with respect to radius.
        # This is ON_DEMAND, so we only grab this when it's requested.
        return sp.simplify(sp.diff(expression, axes[0]))

    def get_limiting_behavior(
        self, limit: str = "inner", strict: bool = True
    ) -> Optional[Tuple[float, float]]:
        r"""
        Extract the coefficient and power of the dominant power-law term
        for the radial profile in the specified limit.

        Parameters
        ----------
        limit : str, optional
            Specifies the limit to analyze. Options are:

            - 'inner': As :math:`r \to 0`.
            - 'outer': As :math:`r \to \infty`.

            Default is 'inner'.
        strict : bool, optional
            If True, raises any exceptions encountered during the calculation.
            If False, suppresses exceptions and returns None on failure.
            Default is True.

        Returns
        -------
        tuple of float or None
            A tuple containing:
            - Coefficient (float): The prefactor of the dominant power-law term.
            - Power (float): The exponent of the dominant power-law term.
            Returns None if the analysis fails and `strict` is False.

        Raises
        ------
        ValueError
            If the limit is invalid or symbolic expansion fails, and `strict` is True.

        Notes
        -----
        This method relies on the utility function :py:func:`get_powerlaw_limit` to perform
        symbolic series expansion and determine the dominant term.

        Examples
        --------

        The Hernquist profile takes the form

        .. math::

            \rho(r) = \frac{\rho_0}{\xi(\xi +1)^3},

        where :math:`\xi = r/r_s`. Now, in the limit as :math:`r \to \infty`, this clearly goes as
        :math:`\rho_0/\xi^{-4}`, while as :math:`r \to 0`, :math:`\rho \to \rho_0/\xi^{-1}`. We should
        obtain the same answer from this method:

        .. plot::

            >>> from pisces.profiles.density import HernquistDensityProfile
            >>> import matplotlib.pyplot as plt
            >>> density_profile = HernquistDensityProfile(rho_0=5,r_s=1)
            >>> _inner_lim = density_profile.get_limiting_behavior(limit='inner')
            >>> _outer_lim = density_profile.get_limiting_behavior(limit='outer')

            Let's now use this information to make a plot of our profile and its limiting behavior:

            >>> fig,axes = plt.subplots(1,1)
            >>> r = np.logspace(-3,3,1000)
            >>> _ = axes.loglog(r,density_profile(r),label=r'$\rho(r)$')
            >>> _ = axes.loglog(r, _outer_lim[0]*(r**_outer_lim[1]), label=r'$\lim_{r \to \infty} \rho(r)$')
            >>> _ = axes.loglog(r, _inner_lim[0]*(r**_inner_lim[1]), label=r'$\lim_{r \to 0} \rho(r)$')
            >>> _ = axes.set_ylim([1e-6,1e5])
            >>> _ = axes.set_xlim([1e-3,1e3])
            >>> _ = axes.set_xlabel('r')
            >>> _ = axes.set_ylabel('density(r)')
            >>> _ = axes.legend(loc='best')
            >>> plt.show()

        """
        from pisces.utilities.math_utils.symbolic import get_powerlaw_limit

        # Validate the symbolic expression and axis
        if not hasattr(self, "symbolic_expression"):
            raise ValueError("The profile lacks a valid symbolic expression.")
        if not self.SYMBAXES or len(self.SYMBAXES) < 1:
            raise ValueError(
                "The profile must define at least one symbolic axis ('r')."
            )

        try:
            # Extract the symbolic radial axis
            r_symbol = self.SYMBAXES[0]
            return get_powerlaw_limit(self.symbolic_expression, r_symbol, limit=limit)
        except Exception as e:
            if strict:
                raise ValueError(f"Failed to determine limiting behavior: {e}")
            return None
