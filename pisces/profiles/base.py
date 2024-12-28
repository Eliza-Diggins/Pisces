"""
Base classes for Pisces profile management.
"""
from abc import ABCMeta, ABC
from typing import TYPE_CHECKING, Callable, Union, List, Optional

import h5py
import numpy as np
from unyt import Unit

from pisces.profiles.registries import _DEFAULT_PROFILE_REGISTRY

if TYPE_CHECKING:
    pass


import sympy as sp
from typing import Type, Dict, Any, Tuple
from functools import wraps

def class_expression(name: str = None):
    """
    Decorator to register a symbolic expression at the class level.

    Parameters
    ----------
    name : str, optional
        The name of the derived expression. If not provided, the function name will be used.

    Returns
    -------
    Callable
        The decorated method.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            return func(cls, *args, **kwargs)

        wrapper.class_expression = True
        wrapper.expression_name = name or func.__name__
        return wrapper
    return decorator

# noinspection PyTypeChecker
class ProfileMeta(ABCMeta):
    """
    A metaclass for managing Profile subclasses with automatic symbolic attribute generation.

    This metaclass handles the following tasks:
    1. Registers concrete subclasses in a specified registry if ``_REGISTER`` is set to ``True``.
    2. Generates symbolic representations (``SYMBAXES``, ``SYMBPARAMS``, ``_func_expr``) for axes,
       parameters, and the class-level ``_function``.
    3. Derive additional symbolic expressions for the class.

    Abstract classes are ignored for both registration and symbolic generation.
    """

    def __new__(mcs: Type['ProfileMeta'], name: str, bases: Tuple[Type, ...], clsdict: Dict[str, Any]) -> Type:
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
        # Identify if the class is abstract or concrete
        is_abstract = clsdict.get("_IS_ABC", False)

        # Create the new class
        cls = super().__new__(mcs, name, bases, clsdict)

        # Handle registration for concrete subclasses
        if not is_abstract:
            mcs._register_class(cls, clsdict)
            mcs._generate_symbolics(cls)
            mcs._register_class_expressions(cls, clsdict)

        return cls

    @staticmethod
    def _register_class(cls: Type['Profile'], clsdict: Dict[str, Any]):
        """Register the class if `_REGISTER` is enabled."""
        _registration_flag = clsdict.get('_REGISTER', False)
        _registry = clsdict.get('_DEFAULT_REGISTRY', _DEFAULT_PROFILE_REGISTRY)

        if _registration_flag and _registry:
            try:
                _registry.register(cls)
            except Exception as e:
                raise ValueError(f"Failed to register class '{cls.__name__}' due to: {e}")

    @staticmethod
    def _generate_symbolics(cls: Type['Profile']):
        """Generate symbolic axes and parameters."""
        AXES = getattr(cls, 'AXES', None)
        PARAMETERS = getattr(cls, 'PARAMETERS', None)
        _func = getattr(cls, '_function', None)

        if not AXES:
            raise ValueError(f"Class attribute `AXES` is missing in '{cls.__name__}'.")
        if not PARAMETERS:
            raise ValueError(f"Class attribute `PARAMETERS` is missing in '{cls.__name__}'.")
        if not callable(_func):
            raise ValueError(f"Class attribute `_function` is missing or not callable in '{cls.__name__}'.")

        cls.SYMBAXES = [sp.Symbol(ax) for ax in AXES]
        cls.SYMBPARAMS = {param: sp.Symbol(param) for param in PARAMETERS}

        try:
            cls._func_expr = _func(*cls.SYMBAXES, **cls.SYMBPARAMS)
        except Exception as e:
            raise ValueError(f"Failed to generate symbolic function for '{cls.__name__}' due to: {e}")

    @staticmethod
    def _register_class_expressions(cls: Type['Profile'], clsdict: Dict[str, Any]):
        """Register class-level expressions from decorated methods."""
        for attr_name, method in clsdict.items():
            if callable(method) and getattr(method, 'class_expression', False):
                expression_name = method.expression_name
                expression = method(cls.SYMBAXES, cls.SYMBPARAMS, cls._func_expr)
                cls.set_class_expression(expression_name, expression)

class Profile(ABC,metaclass=ProfileMeta):
    """
    Abstract base class for mathematical profiles with symbolic and numerical support.

    This class provides a framework for defining mathematical profiles with symbolic
    expressions, parameterized inputs, and units. It supports automatic symbolic representation,
    validation of input parameters, unit handling, and serialization/deserialization.

    .. rubric:: Implementation Details

    Each :py:class:`Profile` class is characterized by only a couple of special class-level attributes:

    - :py:attr:`Profile.AXES`: Provides a list of the axes that can be passed to the profile. These become
      the arguments of the eventual ``callable`` object the user interacts with.
    - :py:attr:`Profile.PARAMETERS`: A dictionary with parameter names as keys and their default values. This
      eventually becomes the ``kwargs`` that the user passes during instantiation.
    - :py:attr:`Profile.UNITS`: A string specifying the default units of the profile. These can be overwritten later, but they
      do define the permissible units as user specified units which are incompatible lead to an error.
    - ``_function``: A class method implementing the symbolic mathematical function of the profile.

    With these 4 attributes defined, the :py:class:`Profile` subclass is fully functional.

    Each :py:class:`Profile` class is also registered to a registry (unless ``_REGISTER = False``), which can
    be specified with ``_DEFAULT_REGISTRY``. The flag ``_IS_ABC`` indicates that we should ignore axes and symbols. This
    should always be ``True`` for abstract classes and ``False`` otherwise.

    .. rubric:: The Metaclass

    Behind the scenes, the :py:class:`ProfileMeta` class drives the construction of each class. This performs the
    following basic operations:

    1. Register the class with the registry.
    2. Convert the string versions of the :py:attr:`Profile.AXES` and :py:attr:`Profile.PARAMETERS` into ``sympy`` symbols.
    3. Pass the symbols from step 2 into ``_function`` to generate an expression.

    Then, once the class is instantiated, we fill the expression with the parameter values given by the user to
    create both a symbolic and numeric form of the function. This relies on the ``lambdify`` protocol from ``sympy``, which
    converts the sympy expression into the callable.


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

    For example:

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

    ### Instance-Level Expressions

    Instance-level expressions are specific to a particular instance and can override or extend the class-level definitions.
    They are managed using the following methods:

    - **Define**: Use the :py:meth:`Profile.set_expression` method to register an instance-level symbolic expression.
    - **Access**: Use the :py:meth:`Profile.get_expression` method to retrieve a registered expression, optionally searching
      the class-level expressions if not found at the instance level.
    - **Evaluate Numerically**: Use the :py:meth:`Profile.get_numeric_expression` method to retrieve a callable numeric version of the expression.

    For example:

    .. code-block:: python

        instance = MyProfile(a=3, b=5)
        instance.set_expression('intercept', instance.substitute_expression('b'))
        intercept_symbolic = instance.get_expression('intercept')
        intercept_numeric = instance.get_numeric_expression('intercept')  # Returns a callable for numerical evaluation

    ### Summary of Expression Management

    | Method                          | Scope        | Purpose                                               |
    |---------------------------------|--------------|-------------------------------------------------------|
    | :py:meth:`set_class_expression` | Class-Level  | Define a symbolic expression at the class level.      |
    | :py:meth:`get_class_expression` | Class-Level  | Retrieve a symbolic expression from the class level.  |
    | :py:meth:`set_expression`       | Instance     | Define a symbolic expression at the instance level.   |
    | :py:meth:`get_expression`       | Instance     | Retrieve a symbolic expression from the instance or, optionally, the class level. |
    | :py:meth:`get_numeric_expression` | Instance   | Retrieve or create a callable numeric version of an expression. |

    By combining these features, the :py:class:`Profile` class provides a powerful and flexible framework for
    modeling mathematical profiles with extensible symbolic and numeric capabilities.

    .. admonition:: Developer Note

        To instruct a class to register and derive a specific derived expression during initialization, the :py:func:``class_expression``
        decorator should be added to a method taking 3 arguments: ``axes, parameters, expression``, where ``axes`` will be :py:attr:`Profile.SYMBAXES`,
        ``parameters`` will be :py:attr:`Profile.SYMBPARAMS`, and ``expression`` will be :py:attr:`Profile._func_expr`. It should return
        a symbolic expression which is then registered in the class level ``_EXPR_DICT`` dictionary.
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _REGISTER = True
    _DEFAULT_REGISTRY = _DEFAULT_PROFILE_REGISTRY
    _IS_ABC = True
    _EXPR_DICT = {}

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = None
    """ list of str: The axes (free parameters) of the profile.
        These should be strings and be specified in order. The metaclass will process
        them into the symbolic axes of :py:attr:`Profile.SYMBAXES`.
    """
    PARAMETERS: Dict[str, Any] = None
    """ dict of str, Any: The parameters of the profile.
        The keys of the dictionary should specify the symbol for the kwarg (its name) and the values
        should represent default values (``float``). These do not possess inherent units, instead, they
        are simply to be provided in a unit consistent way.
    """
    UNITS: str = ""
    """ str: The default units of the profile.
        These can later be overridden by specifying the ``units`` kwarg at instantiation.
    """

    SYMBAXES: List[sp.Symbol] = None
    """ list of sp.Symbol: The symbolic axes of the profile."""
    SYMBPARAMS: Dict[str, sp.Symbol] = None
    """ dict of sp.Symbol, float: The symbolic parameters of the profile."""

    def __init__(self, units: Union[str,'Unit'] =None,**kwargs):
        """
        Initialize the Profile instance with parameter values and units.

        Parameters
        ----------
        units : str or Unit, optional
            Units for the profile, default is the class-level ``UNITS``.
        kwargs :
            Parameter values to use for the profile.

        Raises
        ------
        ValueError
            If provided units are incompatible with the class's base units.
        """
        # Set the parameters.
        self.parameters = self._validate_and_update_params(kwargs)

        # Handle units -- Make sure they are valid when converted to the
        # default. This ensures dimensionality constraints.
        self.units = Unit(units or self.UNITS)

        try:
            _ = self.units.get_conversion_factor(Unit(self.__class__.UNITS))
        except Exception as _:
            raise ValueError(f"Units {self.units} are not consistent with base units for {self.__class__.__name__} ({self.__class__.UNITS}).")

        # Setup the call function.
        self._setup_call_function()

        # Setup the repository for derived symbolics and
        # lambdified methods
        self._derived_symbolics: Dict[str,sp.Basic] = {}
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
        parameters = self.__class__.PARAMETERS.copy()

        # Update with user-provided parameters
        for key, value in kwargs.items():
            if key in parameters:
                parameters[key] = value
            else:
                raise ValueError(f"Unexpected parameter: '{key}' not recognized in class '{self.__class__.__name__}'.")

        # Check for missing required parameters
        missing = [key for key, value in parameters.items() if value is None]
        if missing:
            raise ValueError(f"Missing required parameters for class '{self.__class__.__name__}': {missing}")

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
        Substitute the parameter values into a symbolic expression.

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
        return sp.lambdify(self.__class__.SYMBAXES, expression, 'numpy')

    def __call__(self, *args) -> Any:
        """
        Evaluate the profile function with given inputs.

        Parameters
        ----------
        *args : tuple
            Input values for the independent variables.

        Returns
        -------
        Any
            Evaluated result of the profile function.
        """
        return self._call_function(*args)

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"<{self.__class__.__name__}({param_str})>"

    def __str__(self) -> str:
        return self.__repr__()

    def to_hdf5(self, h5_obj: Union[h5py.File, h5py.Group], group_name: str, overwrite: bool = False):
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
                raise ValueError(f"Group '{group_name}' already exists. Use `overwrite=True` to replace it.")

        group = h5_obj.create_group(group_name)
        group.attrs["class_name"] = self.__class__.__name__
        group.attrs["axes"] = np.array(self.__class__.AXES, dtype='S')
        group.attrs["units"] = str(self.units)

        for key, value in self.parameters.items():
            group.attrs[key] = value

    @classmethod
    def from_hdf5(cls, h5_obj: Union[h5py.File, h5py.Group], group_name: str) -> "Profile":
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
        if class_name != cls.__name__:
            raise ValueError(
                f"Mismatch in class name. Expected '{cls.__name__}', found '{class_name}' in group '{group_name}'."
            )

        # Load parameters and validate
        parameters = {key: group.attrs[key] for key in group.attrs if key not in ["class_name", "axes", "units"]}
        return cls(units=group.attrs["units"], **parameters)

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
    def set_class_expression(cls, expression_name: str, expression: sp.Basic, overwrite: bool = False):
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
            raise ValueError(f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it.")
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
        try:
            return cls._EXPR_DICT[expression_name]
        except KeyError:
            raise KeyError(f"Class-level expression '{expression_name}' not found.")

    def get_expression(self, expression_name: str, search_class: bool = False) -> sp.Basic:
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
                return self.substitute_expression(self.get_class_expression(expression_name))
            except KeyError:
                pass

        raise KeyError(f"Expression '{expression_name}' not found.")

    def set_expression(self, expression_name: str, expression: sp.Basic, overwrite: bool = False):
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
        if ((expression_name in self._derived_symbolics) or (expression_name in self._EXPR_DICT)) and not overwrite:
            raise ValueError(f"Expression '{expression_name}' already exists. Use `overwrite=True` to replace it.")
        self._derived_symbolics[expression_name] = expression

    def get_numeric_expression(self, expression_name: str, search_class: bool = False) -> Callable:
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
            symbolic_expression = self.get_expression(expression_name, search_class=search_class)
            self._derived_numerics[expression_name] = self.lambdify_expression(symbolic_expression)
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
    AXES: List[str] = ['r']
    """
    list of str: The axes (free parameters) of the profile.
        These should be strings and specified in order. The metaclass processes
        them into symbolic axes via :py:attr:`Profile.SYMBAXES`.
    """
    def get_limiting_behavior(self, limit: str = 'inner', strict: bool = True) -> Optional[Tuple[float, float]]:
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
        if not hasattr(self, 'symbolic_expression'):
            raise ValueError("The profile lacks a valid symbolic expression.")
        if not self.SYMBAXES or len(self.SYMBAXES) < 1:
            raise ValueError("The profile must define at least one symbolic axis ('r').")

        try:
            # Extract the symbolic radial axis
            r_symbol = self.SYMBAXES[0]
            return get_powerlaw_limit(self.symbolic_expression, r_symbol, limit=limit)
        except Exception as e:
            if strict:
                raise ValueError(f"Failed to determine limiting behavior: {e}")
            return None