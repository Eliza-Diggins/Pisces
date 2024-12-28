r"""
Coordinate Systems for Pisces
=============================

This module contains the coordinate systems for the Pisces library. This includes various spheroidal
coordinate systems, cartesian coordinate systems, cylindrical coordinate systems, and others. For each of
these coordinate systems, structures have been generated to allow for differential operations and the tracking
of symmetries through those operations.

Mathematical Background
=======================

An orthogonal coordinate system is defined by a set of coordinate surfaces that intersect
at right angles. Each coordinate axis has an associated Lame coefficient, :math:`h_i`, which
scales differential elements along that axis. Orthogonal systems are useful in physics and
engineering because they allow the calculation of differential operators (e.g., gradient,
divergence, and Laplacian) based on Lame coefficients, simplifying the complexity of vector
calculus in curved spaces.

A more complete overview of the relevant theory may also be found here :ref:`geometry_theory`.

Basis Vectors
-------------

Orthogonal coordinate systems feature a set of **basis vectors** for each point in space, which
are tangent to the curves obtained by varying a single coordinate while holding others constant.
Unlike in Cartesian coordinates where basis vectors are constant, these basis vectors vary across
points in general orthogonal coordinates but remain mutually orthogonal.

- **Covariant Basis** : The covariant basis vectors :math:`\mathbf{e}_i` align with the coordinate
  curves and are derived as:

  .. math::

     \mathbf{e}_i = \frac{\partial \mathbf{r}}{\partial q^i}

  where :math:`\mathbf{r}` is the position vector and :math:`q^i` represents the coordinates.
  These vectors are not typically unit vectors and can vary in length.

- **Unit Basis** : By dividing the covariant basis by their **scale factors**, the normalized
  basis vectors :math:`\hat{\mathbf{e}}_i` are obtained:

  .. math::

     \hat{\mathbf{e}}_i = \frac{\mathbf{e}_i}{h_i}

  .. note::

      These **scale factors** are also referred to as **Lame Coefficients**. That is the term used
      in Pisces. They are a critical component of all of the vector calculus computations that occur in
      a given geometry.

- **Contravariant Basis**: The contravariant basis vectors, denoted :math:`\mathbf{e}^i`, are reciprocal
  to the covariant basis and provide a means to express vectors in a dual basis. In orthogonal systems, they are
  simply related by reciprocal lengths to the covariant vectors:

  .. math::

     \mathbf{e}^i = \frac{\hat{\mathbf{e}}_i}{h_i} = \frac{\mathbf{e}_i}{h_i^2}

  These satisfy the orthogonality condition:

  .. math::

     \mathbf{e}_i \cdot \mathbf{e}^j = \delta_i^j

  where :math:`\delta_i^j` is the Kronecker delta.

  .. hint::

      For those with a mathematical background, the contravariant basis forms the **dual basis** to the
      covariant basis at a given point. Thus, in some respects, these vectors actually live in entirely different
      spaces; however, this detail is of minimal importance in this context.

Differential Operators
----------------------

Let :math:`\phi` be a scalar field in a space equipped with a particular coordinate system. Evidently, the differential
change in :math:`\phi` is

.. math::

    d\phi = \partial_i \phi dq^i

The gradient is, by definition, an operator such that :math:`d\phi = \nabla \phi \cdot d{\rm r}`. Clearly,

.. math::

    d{\bf r} = \partial_i {\bf r} dq^i = {\bf e}_i dq^i,

so

.. math::

    \nabla_k \phi \cdot d{\rm r} = \nabla_{kj} {\bf e}_j dq^j = \partial_k \phi dq^k.

Thus,

.. math::

    \nabla_{kj} \phi = \partial_k \phi e^k \implies \nabla \phi \cdot d{\rm r} = \partial_k \phi dq^k e^k \cdot e_k = \partial_k \phi dq^k.

As such, the general form of the gradient for a scalar field is

.. math::

   \nabla \phi = \sum_{i} \frac{\hat{\mathbf{e}}_i}{h_i} \frac{\partial \phi}{\partial q^i}

More in-depth analysis is needed to understand the divergence, curl, and Laplacian; however, the results take the following form:

- **Divergence of a vector field** : For a vector field :math:`\mathbf{F} = \sum_{i} F_i \hat{\mathbf{e}}_i`,
  the divergence is computed as:

  .. math::

     \nabla \cdot \mathbf{F} = \frac{1}{\prod_k h_k} \sum_{i} \frac{\partial}{\partial q^i} \left( \frac{\prod_k h_k}{h_i}\mathbf{F}_i \right)

- **Laplacian of a scalar field** : For a scalar field :math:`\phi`, the Laplacian is given by:

  .. math::

     \nabla^2 \phi = \frac{1}{\prod_k h_k} \sum_{i} \frac{\partial}{\partial q^i} \left( \frac{\frac{1}{\prod_k h_k}}{h_i^2} \frac{\partial \phi}{\partial q^i} \right)

The notation may be made simpler by introducing the **Jacobian** determinant:

.. math::

    J = \prod_i h_i.

In this case, we then find

- **Divergence of a vector field** : For a vector field :math:`\mathbf{F} = \sum_{i} F_i \hat{\mathbf{e}}_i`,
  the divergence is computed as:

  .. math::

     \nabla \cdot \mathbf{F} = \frac{1}{J} \sum_{i} \frac{\partial}{\partial q^i} \left( \frac{J}{h_i}\mathbf{F}_i \right)

- **Laplacian of a scalar field** : For a scalar field :math:`\phi`, the Laplacian is given by:

  .. math::

     \nabla^2 \phi = \frac{1}{J} \sum_{i} \frac{\partial}{\partial q^i} \left( \frac{J}{h_i^2} \frac{\partial \phi}{\partial q^i} \right)

See Also
--------
`Orthogonal coordinates <https://en.wikipedia.org/wiki/Orthogonal_coordinates>`_

"""
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, List, Dict, TYPE_CHECKING, Callable, Union, Optional

import numpy as np
import sympy as sp

from pisces.geometry.exceptions import ConversionError
from pisces.geometry.utils import CoordinateConverter
from pisces.utilities.array_utils import CoordinateArray
from pisces.utilities.general import find_in_subclasses
from utilities.math_utils.numeric import partial_derivative, function_partial_derivative
from pisces.utilities.logging import mylog

if TYPE_CHECKING:
    from pisces.geometry._typing import AxisAlias


# noinspection PyProtectedMember,PyUnresolvedReferences
class CoordinateSystemMeta(ABCMeta):
    # Meta class that composes the sympy components of the coordinate system
    # class, simplifies, etc. to construct the relevant coordinate system.
    _IGNORED = ['CoordinateSystem','RadialCoordinateSystem']
    def __new__(mcs, name, bases, namespace, **kwargs):
        if name not in mcs._IGNORED:
            # @ SYMBOL CONSTRUCTION @ #
            # The metaclass now processes the axes and parameters from the namespace to
            # produce the relevant symbols in SYMBAXES and SYMBPARAMS
            mcs._construct_axes_symbols(namespace)
            mcs._construct_parameter_symbols(namespace)

            # @ PROCESS LAME COEFFICIENT FUNCTIONS @ #
            # We now process the lame functions. As it stands, they are functions of
            # the symbols and return sympy expressions. We need to validate them and then
            # process them to lambda functions in the namespace.
            mcs._process_lame_coefficients(namespace)

        return super().__new__(mcs, name, bases, namespace, **kwargs)

    @classmethod
    def _construct_axes_symbols(mcs, namespace):
        # This private method constructs the symbols for each of the
        # axes in the coordinate system. It should NOT be altered in
        # subclasses.
        if 'AXES' not in namespace:
            raise ValueError("AXES not in namespace")

        _axes = namespace['AXES']
        namespace['SYMBAXES'] = [sp.Symbol(ax) for ax in _axes]

    @classmethod
    def _construct_parameter_symbols(mcs, namespace):
        # This private method constructs the symbols for each of the
        # parameters in the coordinate system. It should NOT be altered in
        # subclasses.
        if 'PARAMETERS' not in namespace:
            raise ValueError("PARAMETERS not in namespace")

        _parameters = list(namespace['PARAMETERS'].keys())
        namespace['SYMBPARAMS'] = {param: sp.Symbol(param) for param in _parameters}

    @classmethod
    def _process_lame_coefficients(mcs, namespace):
        # Identify the Lame Coefficient functions in the class. This could
        # be modified to achieve a different scheme for identifying Lame coefficients.
        #
        # Currently, we seek out the lame coefficients based on name and raise
        # errors if the function is missing from the class.
        #
        # Pull the number of dimensions so we know how many Lame coefficients there are.
        try:
            cs_ndim = namespace['NDIM']
        except KeyError:
            raise ValueError("Failed to determine `NDIM` while processing Lame coefficients. Did you"
                             " forget to add NDIM to a subclass of CoordinateSystem?")

        # Start pulling the lame coefficients from the name space and sympifying them.
        lame_coefficients = {}

        for dim in range(cs_ndim):
            cf_attr = namespace.pop(f'lame_{dim}', None) # ! REMOVES lame_dim from the namespace.
            if cf_attr is None:
                raise ValueError(f"Failed to locate `lame_{dim}` while processing Lame coefficients. Did you"
                                 f" forget to implement it for a subclass of CoordinateSystem?")

            lame_coefficients[dim] = cf_attr

        # obtain the relevant parameters and axes, then construct the _lame_symbolic attribute.
        PARAMS = namespace['SYMBPARAMS']
        AXES = namespace['SYMBAXES']
        namespace['_lame_symbolic'] = {namespace['AXES'][dim]: sp.sympify(lcoeff_func(*AXES, **PARAMS)) for
                                       dim, lcoeff_func in lame_coefficients.items()}

# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker
class CoordinateSystem(ABC, metaclass=CoordinateSystemMeta):
    """
    Abstract base class representation of an orthogonal coordinate system.

    This class serves as a base for specific orthogonal coordinate systems, defining key
    mathematical methods to compute differential operators, transformations, and basis
    functions. It utilizes a ``sympy`` based backend to compute critical coefficients for differential
    operators and then ``lambdify``-s them into numpy function for rapid execution.

    Attributes
    ----------
    NDIM : int
        Number of dimensions in the coordinate system.
    AXES : list of str
        Names of the coordinate system axes, defaulting to ``['x', 'y', 'z']``.

    Notes
    -----
    An orthogonal coordinate system is defined by a set of coordinate surfaces that intersect
    at right angles. Each coordinate axis has an associated Lame coefficient, :math:`h_i`, which
    scales differential elements along that axis.

    Orthogonal coordinate systems allow the computation of fundamental operators (gradient,
    divergence, and Laplacian) based on Lame coefficients.

    """
    # @@ CLASS ATTRIBUTES @@ #
    # DEVELOPERS: these should be set in any subclass of CoordinateSystem. The axes
    #  should follow standard convention and the parameters and axes will become the
    #  symbolic attributes of the class.
    NDIM: int = 3
    AXES: list[str] = ['x','y','z']
    PARAMETERS: Dict[str, Any] = {}
    _SKIP_LAMBDIFICATION = []

    # @@ DYNAMICALLY GENERATED ATTRIBUTES @@ #
    # These attributes are generated dynamically in the metaclass
    # and then populated based on other class methods.
    #
    # DEVELOPERS: If changes are made to the metaclass, new variables could be added here
    #  to ensure that IntelliJ understands what's going on.
    SYMBAXES: List[sp.Symbol] = None
    SYMBPARAMS: Dict[str, sp.Symbol] = None
    _lame_symbolic: Dict[str, sp.Basic] = None

    # @@ INITIALIZATION @@ #
    # The initialization of the CoordinateSystem class is broken down into submethods
    # to permit easy overwriting for developers. Subclasses may re-implement any of the
    # methods below as needed.
    def __init__(self, **kwargs):
        # @ PARSE parameters @ #
        # Convert kwargs to self.parameters attribute.
        self._setup_parameters(**kwargs)

        # @ SETUP lame coefficients @ #
        # This step converts the symbolic Lame coefficients to
        # the equivalent numpy functions.
        self._symbolic_attributes = {}
        self._lambdified_attributes: Dict[str, Callable] = {}
        self._setup_lame_coefficients()

        # @ DERIVED properties setup @ #
        # The _derive_symbolic_attributes method now fills the symbolic
        # attributes dictionary with the relevant symbolic components.
        mylog.info(
            f"Preparing derived attributes for {self.__class__.__name__} instance. This may take a few seconds...")
        self._derive_symbolic_attributes()

        # Lambdifying symbolic attributes
        self._lambdify_derived_attributes()

    def _setup_parameters(self, **kwargs):
        self.parameters = self.__class__.PARAMETERS.copy()
        self.parameters.update(kwargs)

        if any(kwarg not in self.PARAMETERS for kwarg in self.parameters):
            raise ValueError("Unknown parameter '%s'" % self.parameters)

    def _setup_lame_coefficients(self):
        """
        Prepares the Lame coefficient functions by binding parameters and lambdifying.
        Ensures that constants are converted to array-compatible expressions.
        """
        self._lame_functions = []
        self._lame_inst_symbols = {}

        mylog.info(f"Preparing Lame Coefficients for {self.__class__.__name__} instance. This may take a few seconds...")
        for ax in self.AXES:
            mylog.debug(f"\t [COMPLETE] Axis {ax} finished.")
            symbolic_lame_function = self.__class__._lame_symbolic[ax]
            self._lame_inst_symbols[ax] = sp.simplify(symbolic_lame_function.subs(self.parameters))
            self._lame_functions.append(self.lambdify_expression(symbolic_lame_function))

    def _derive_symbolic_attributes(self):
        self._symbolic_attributes['jacobian'] = sp.simplify(sp.prod([self.get_lame_symbolic(ax) for ax in self.AXES]).subs(
            self.parameters))
        mylog.debug(f"\t [COMPLETE] Derived the jacobian...")

    def _lambdify_derived_attributes(self):
        mylog.info("Lambdifying attributes...")
        for sym_attr_name, symb_attr_value in self._symbolic_attributes.items():
            if sym_attr_name in self.__class__._SKIP_LAMBDIFICATION:
                mylog.debug(f"\t [SKIP] Skipping {sym_attr_name}")
                continue

            self._lambdified_attributes[sym_attr_name] = self.lambdify_expression(symb_attr_value)
            mylog.debug(f"\t [COMPLETE] Lambdifyed attribute {sym_attr_name}...")

    def lambdify_expression(self, expression: Union[str, sp.Basic]) -> Callable:
        """
        Convert an expression into a callable numpy function using the attributes and
        coordinates of this coordinate system.

        Parameters
        ----------
        expression: str or sp.Basic
            The sympy expression or string corresponding to the function to be evaluated.

        Returns
        -------
        Callable
            The numpy function corresponding to the expression.

        Notes
        -----
        Errors will occur when the expression cannot be converted. This most often occurs because the input
        expression contains variables which are not recognized / understood in the coordinate system.
        """
        # COERCE the expression to a sympy expression and then bind it to the constants.
        expression = sp.sympify(expression)
        bound_expression = expression.subs(self.parameters)

        # Handle the constant case by constructing a lambda function.
        if bound_expression.is_constant():  # Check if the expression is constant
            # Convert constant to a numerical value
            constant_value = float(sp.simplify(bound_expression))

            # Create a lambda function returning an array of the same shape as the first argument
            return lambda *args, cv=constant_value: np.full_like(args[0], cv)
        else:
            return sp.lambdify(self.SYMBAXES, bound_expression, 'numpy')

    # @@ DERIVED ATTRIBUTE METHODS @@ #
    # For each of the attributes in _derive_symbolic_attributes, the symbolic calculation
    # should either occur directly in _derive_symbolic_attributes (if its unlikely to need changing)
    # or in a seperate method below (if it might need to be changed).
    #
    # DEVELOPER NOTE: These generally do NOT need to be changed in subclasses, but may be changed
    #  if the developer so needs.
    def get_derived_attribute_symbolic(self, attribute_name: str) -> sp.Basic:
        try:
            return self._symbolic_attributes[attribute_name]
        except KeyError:
            raise ValueError(
                f'Attribute `{attribute_name}` is not defined symbolically for class `{self.__class__.__name__}`.')

    def get_derived_attribute_function(self, attribute_name: str) -> Callable:
        try:
            return self._lambdified_attributes[attribute_name]
        except KeyError:
            raise ValueError(
                f'Attribute `{attribute_name}` is not defined symbolically for class `{self.__class__.__name__}`.')

    def _eval_der_attr_func(self, attribute_name: str, coordinates: np.ndarray) -> np.ndarray:
        coordinates = CoordinateArray(coordinates, self.NDIM)
        coordinates = np.moveaxis(coordinates, -1, 0)

        func = self.get_derived_attribute_function(attribute_name)
        return func(*coordinates)

    def jacobian(self, coordinates: np.ndarray) -> np.ndarray:
        r"""
        Compute the Jacobian for the given coordinate system at specified points.

        The Jacobian is calculated as the product of the Lame coefficients across all axes at each
        point in the coordinate system, representing the volume scaling factor for the transformation
        from Cartesian coordinates to the current coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shape ``(*, NDIM)``, where ``*`` may represent any generic array structure and
            ``NDIM`` is the number of dimensions in the coordinate system.

        Returns
        -------
        NDArray
            The Jacobian determinant values at each specified point, with shape ``(P,)``. Each
            value represents the volume scaling factor at that point.

        Notes
        -----
        In non-Cartesian coordinate systems, the Jacobian determinant is given by the product of the
        Lame coefficients, :math:`h_1 h_2 \dots h_N`, where each :math:`h_i` is the Lame coefficient
        for the ``i``-th axis. The Jacobian determinant is critical in volume integrations and changes
        of variables in non-Cartesian coordinates.
        """
        return self._eval_der_attr_func('jacobian', coordinates)

    def get_symbolic_D_term(self, axis: 'AxisAxis', basis: str):
        # VALIDATION: construct the axis index and the d_term name.
        axis_index = self.ensure_axis_numeric(axis)
        D_name = f"D_{axis_index}_{basis}"

        # CONSTRUCT the symbol first.
        if D_name not in self._symbolic_attributes:
            _scale = dict(unit=-1, contravariant=-2, covariant=0)[basis]
            _lame = self.get_lame_symbolic(axis)
            J = self.get_derived_attribute_symbolic('jacobian')
            D = sp.simplify((1 / J) * sp.diff((J / _lame ** _scale), self.SYMBAXES[axis_index]))
            self._symbolic_attributes[D_name] = D

        return self._symbolic_attributes[D_name]


    def D_term(self, coordinates: np.ndarray, axis: 'AxisAlias', basis: str) -> np.ndarray:
        r"""
        Computes the :math:`D`-term for the divergence along a given axis in a particular basis.

        Parameters
        ----------
        coordinates: NDArray
            The coordinates at which to evaluate the :math:`D`-term. These should be ``(...,NDIM)`` in shape where
            ``NDIM`` is the number of dimensions in the coordinate system.
        axis: str or int
            The axis (or index) for which to compute the :math:`D`-term.
        basis: str
            The basis in which to compute the :math:`D`-term. This may be ``"unit"``, ``"covariant"`` or ``"contravariant"``.

        Returns
        -------
        NDArray
            The value of the :math:`D`-term at each point in the coordinates. This will be ``(...,)`` in shape.

        Notes
        -----

        In orthogonal coordinates, the divergence of a vector field :math:`{\bf F}` is

        .. math::

            \nabla \cdot {\bf F} = \frac{1}{J}\partial_{k} \left(\frac{J}{\lambda_k} \hat{\bf F}_k\right),

        where :math:`\hat{\bf F}_k` is the **unit-basis** component of :math:`{\bf F}`. Employing the product rule,

        .. math::

            \nabla \cdot {\bf F} = \underbrace{\frac{1}{J} \partial_k \left(\frac{J}{\lambda_k} \right)}_{\hat{D}_k} \hat{\bf F}_k + \frac{1}{\lambda k} \partial_k \hat{\bf F}_k,

        where :math:`\hat{D}` is the :math:`D`-term of the divergence in the unit basis. An equivalent (with additional scaling in the derivative) may be constructed
        for each of the other bases. Thus, the divergence simplifies to

        .. math::

            \nabla \cdot {\bf F} = \hat{D}_k \hat{\bf F}_k + \lambda_k^{-1} \partial_k \hat{\bf F}_k.

        """
        axis_index = self.ensure_axis_numeric(axis)
        D_name = f"D_{axis_index}_{basis}"
        # CONSTRUCT the lambda function
        if D_name not in self._lambdified_attributes:
            self._lambdified_attributes[D_name] = self.lambdify_expression(self.get_symbolic_D_term(axis, basis))
        return self._eval_der_attr_func(D_name, coordinates)

    def get_symbolic_L_term(self, axis: 'AxisAlias'):
        return self.get_symbolic_D_term( axis, basis='contravariant')

    def L_term(self, coordinates: np.ndarray, axis: 'AxisAlias') -> np.ndarray:
        r"""
        Computes the :math:`L`-term for the Laplacian along a given axis.

        Parameters
        ----------
        coordinates: NDArray
            The coordinates at which to evaluate the :math:`L`-term. These should be ``(...,NDIM)`` in shape where
            ``NDIM`` is the number of dimensions in the coordinate system.
        axis: str or int
            The axis (or index) for which to compute the :math:`L`-term.

        Returns
        -------
        NDArray
            The value of the :math:`L`-term at each point in the coordinates. This will be ``(...,)`` in shape.

        Notes
        -----
        The Laplacian of a scalar field :math:`f` in orthogonal coordinates is given by

        .. math::

            \nabla^2 f = \frac{1}{\lambda_k^2} \partial_k^2 f + \underbrace{\frac{1}{J} \partial_k \left( \frac{J}{\lambda_k^2} \right)}_{L_k} \partial_k f,

        where :math:`\lambda_k` is the Lame coefficient for the :math:`k`-th axis, and :math:`J` is the Jacobian determinant.
        The :math:`L`-term, :math:`L_k`, is defined as

        .. math::

            L_k = \frac{1}{J} \partial_k \left( \frac{J}{\lambda_k^2} \right).

        This term accounts for the contribution to the Laplacian from the coordinate system's geometry.

        In practice, the Laplacian simplifies to:

        .. math::

            \nabla^2 f = L_k \partial_k f + \frac{1}{\lambda_k^2} \partial_k^2 f.

        The :math:`L`-term is specific to the contravariant basis.
        """
        return self.D_term(coordinates, axis, 'contravariant')

    # @@ DIFFERENTIAL OPERATORS @@ #
    # These are the core methods for evaluating differential operations. They generally
    # do not need to be altered.
    def compute_derivative(self,
                           field: Union[np.ndarray, Callable],
                           coordinates: np.ndarray,
                           axes: Union['AxisAlias',List['AxisAlias']],
                           **kwargs) -> np.ndarray:
        r"""
        Compute the derivative of a scalar field along a specific axis.

        Parameters
        ----------
        field : Union[np.ndarray, Callable]
            The field to differentiate:
            - A numpy array for numerical input. This must be a ``(...,)`` array matching the shape of the
              ``coordinates`` argument (up to the final dimension).
            - A callable function returning a numpy array for functional input. This should have signature
              ``f(x_0,...,x_1)`` and return an array of the same shape as each of the input coordinates.
        coordinates : np.ndarray
            Array of coordinates with shape ``(..., NDIM)``, where ``NDIM`` is the number of dimensions.
        axes : str or int or List[str] or List[int]
            The axes along which to compute the derivative. A single axis (or axis index) can be used or a
            number of them may be provided.
        **kwargs
            Additional keyword arguments for numerical differentiation.

        Returns
        -------
        np.ndarray
            The derivative of the field along the specified axis, with shape ``(...,len(axes))``.

        Notes
        -----
        The derivative of a field :math:`f` along axis :math:`k` is defined as:

        .. math::

            \frac{\partial f}{\partial x_k}.

        For callable fields, the derivative is computed using finite differences or an equivalent numerical method.
        For numpy arrays, partial derivatives are computed directly based on the provided axis.

        This function ensures compatibility with both array-like and callable field representations.
        """
        # Validate and reshape coordinates for numerical differentiation
        coordinates = CoordinateArray(coordinates, self.NDIM) # (...,NDIM)

        # Determine the type of the field and validate inputs
        if isinstance(field, np.ndarray):
            field_type = 'array'
        elif callable(field):
            field_type = 'function'
        else:
            raise ValueError(
                f"Unsupported field type: {type(field)}. Expected np.ndarray, or Callable.")

        # Convert axis to numeric if provided as a string
        if not issubclass(axes, (tuple, list)):
            axes = [axes]
        axes_indices = [self.ensure_axis_numeric(axis) for axis in axes]

        # Compute derivative for numerical fields (array or function)
        if field_type == 'array':
            return partial_derivative(coordinates, field, axes=axes_indices, **kwargs)
        elif field_type == 'function':
            return function_partial_derivative(field, coordinates, axes=axes_indices, **kwargs)

        # This return is redundant due to error checks, but added for clarity
        raise RuntimeError("Unexpected execution path in `compute_derivative`.")

    # @ GRADIENT @ #
    def compute_gradient(self,
                         field: Union[np.ndarray, Callable],
                         coordinates: np.ndarray,
                         /,
                         axes: Union['AxisAlias', List['AxisAlias']] = None,
                         *,
                         derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                         basis: str = 'unit',
                         **kwargs) -> np.ndarray:
        r"""
        Compute the gradient of a scalar field.

        Parameters
        ----------
        field : Union[np.ndarray, Callable]
            The scalar field to take the gradient of:
            - A numpy array for numerical input. This must be a ``(...,)`` array matching the shape of the
              ``coordinates`` argument (up to the final dimension).
            - A callable function returning a numpy array for functional input. This should have signature
              ``f(x_0,...,x_1)`` and return an array of the same shape as each of the input coordinates.

            .. warning::

                If derivatives are **not** provided explicitly, then fields passed as ``np.ndarray`` must have a more
                stringent grid shape of ``(N_1,N_2,...,N_NDIM,)`` in order to compute the necessary derivatives.

        coordinates : np.ndarray
            Array of coordinates with shape ``(..., NDIM)``, where ``NDIM`` is the number of dimensions. If numerical
            derivatives are necessary, then ``(..., NDIM)`` must be a grid with more stringent shape ``(N_1,...,N_NDIM,N_DIM)``,
            where ``N_i`` may be any number of grid points.
        axes : Union['AxisAlias', List['AxisAlias']]
            The axes along which to compute the gradient.
        derivatives : Union[np.ndarray,List[Callable]], optional
            Known derivatives along specific axes to be used instead of numerical differentiation. If the ``field``
            is specified as a ``callable``, then ``derivatives`` should be a ``list[Callable]`` providing each of the
            derivatives to be computed (length of ``axes``). If the field is an array, ``derivatives`` should be an array
            matching the shape of the ``coordinates`` up to the final axis and then having ``len(axes)`` elements in the final
            index.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            The basis in which to compute the gradient. Default is 'unit'.
        **kwargs
            Additional keyword arguments for numerical differentiation.

        Returns
        -------
        np.ndarray
            The gradient of the scalar field, with shape ``(..., NDIM)``.

        Notes
        -----
        The gradient of a scalar field :math:`f` in orthogonal coordinates is given by:

        .. math::

            \nabla f = \left( \frac{\partial f}{\partial x_k} \right) \hat{\mathbf{e}}_k,

        where :math:`\hat{\mathbf{e}}_k` are the basis vectors. Depending on the specified basis, scaling factors
        from the Lame coefficients are applied:

        - **Unit basis**: Scaling by :math:`\lambda_k`.
        - **Covariant basis**: Scaling by :math:`\lambda_k^2`.
        - **Contravariant basis**: No scaling.

        The gradient components are returned in the specified basis.
        """
        # VALIDATE inputs.
        # Coordinates are coerced to CoordinateArray, which ensures a shape at least of (...,NDIM).
        # This does not enforce a grid shape -- must be done when derivatives are assessed.
        coordinates = CoordinateArray(coordinates, self.NDIM)

        # Construct the axes and their indices. Fill in if values are not provided.
        if axes is None:
            axes = self.AXES[:]
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        axes = [self.ensure_axis_string(axis) for axis in axes]
        axes_indices = [self.ensure_axis_numeric(axis) for axis in axes]

        # The field type is determined based on type and then (if numpy array), we coerce the field further to
        # match the shape of the coordinates.
        if isinstance(field, np.ndarray):
            field_type = 'array'

            # checking that the field has the correct shape. This will ensure that
            # (..., NDIM) matches the shape of the coordinates.
            try:
                field = field.reshape(coordinates.shape[:-1])
            except Exception as e:
                raise ValueError(f"Field (shape={field.shape}) does not match the shape of coordinates (shape={coordinates.shape}).\n"
                                 f"The field should have had shape {coordinates.shape[:-1]}.\n"
                                 f"ERROR={e}")

        elif callable(field):
            field_type = 'function'
        else:
            raise ValueError(
                f"Unsupported field type: {type(field)}. Expected str, sympy.Basic, np.ndarray, or Callable.")

        # MANAGE the derivatives. If the user provided them, we just need to ensure that they are valid. If not, they
        # need to be computed and we need to enforce further restrictions on the grid shape.
        if derivatives is not None:
            if field_type == 'function':
                # validate that the derivatives are provided correctly and that we have as many as
                # necessary.
                if not isinstance(derivatives, (list, tuple)):
                    raise ValueError(f"Field was specified as a function, but derivatives were provided with type {type(derivatives)}.\n"
                                     "We always expect a list of derivative functions when `field` is a function.")

                if len(derivatives) != len(axes):
                    raise ValueError(f"Computing the gradient over {len(axes)} axes, but the user provided {len(derivatives)} derivatives, which doesn't match.")

                # Coerce the coordinates and then evaluate them.
                # The derivatives should now have shape (...,len(axes))
                d_coords = np.moveaxis(coordinates,-1,0)
                derivatives = np.stack([d(*d_coords) for d in derivatives], axis=-1)

            # validate
            if derivatives.shape != (*coordinates.shape[:-1], len(axes)):
                raise ValueError(f"Derivatives were found to have shape {derivatives.shape} but shape should have been {(*coordinates.shape[:-1], len(axes))}... "
                                 f" Please ensure that you have provided the `derivatives` argument correctly. Consult the documentation for detailed information.")

        else:
            # The derivatives are not provided to us; they need to be computed which will require additional enforced
            # structure on the grid.
            if field_type == 'function':
                derivatives = function_partial_derivative(field, coordinates, axes=axes_indices, **kwargs)
            else:
                # validate the grid shape.
                if coordinates.ndim != self.NDIM + 1:
                    raise ValueError(f"Coordinates must be a grid for numerical derivative computations. Expected {self.NDIM +1} dimensional coordinate array.")
                derivatives = partial_derivative(coordinates, field, axes=axes_indices, **kwargs)

        # FINISH the computation in the correct basis.
        if basis == 'contravariant':
            return derivatives
        elif basis == 'unit':
            return self.scale_by_lame(coordinates, derivatives, axes=axes, order=-1)
        elif basis == 'covariant':
            return self.scale_by_lame(coordinates, derivatives, axes=axes, order=-2)
        else:
            raise ValueError(f"Invalid basis '{basis}'. Expected 'unit', 'covariant', or 'contravariant'.")

    # @ DIVERGENCE @ #
    def compute_divergence(self,
                           field: Union[np.ndarray, Callable],
                           coordinates: Optional[np.ndarray],
                           /,
                           axes: Union['AxisAlias', List['AxisAlias']] = None,
                           *,
                           derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                           basis: str = 'unit',
                           **kwargs) -> np.ndarray:
        r"""
        Compute the divergence of a vector field in the coordinate system.

        Parameters
        ----------
        field : Union[np.ndarray, Callable]
            The vector field to compute the divergence of:
            - A numpy array for numerical input. This must be a ``(...,axes)`` array matching the shape of the
              ``coordinates`` argument (up to the final dimension) and then the number of axes specified.
            - A callable function returning a numpy array for functional input. This should have signature
              ``f(x_0,...,x_1)`` and return an array of shape ``(...,axes)`` as output.
        coordinates : np.ndarray
            The coordinates over which the divergence should be computed. In general, these may be ``(...,NDIM)`` in shape;
            however, if ``derivatives = None`` and the ``field`` is an ``np.ndarray``, then the coordinates must be a proper
            coordinate grid with generic shape ``(N_1,...,N_NDIM, NDIM)`` where each ``N_i`` may be any integer corresponding
            to the number of points along the grid in that dimension.
        axes : Union['AxisAlias', List['AxisAlias']]
            The axes along which the field components are defined.
        derivatives : Union[np.ndarray,List[Callable]], optional
            Known derivatives along specific axes to be used instead of numerical differentiation. The following types
            are supported:
            - (``field`` is ``callable``) The derivatives must be specified as a ``list`` of ``callable`` functions,
              each matching the signature of the ``field``, but returning a **SCALAR** array ``(...,)``. Each element in the
              list should represent the derivative :math:`\partial_k F^k`. No cross terms should be provided.
            - (``field`` is ``np.ndarray``) The derivatives must be specified as a ``np.ndarray`` of shape ``(...,axes)``,
              with each of the final axes corresponding to the :math:`\partial_k F^k` along one of the ``axes``.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            The basis in which the input field is defined. Default is 'unit'.
        **kwargs
            Additional keyword arguments for numerical differentiation.

        Returns
        -------
        np.ndarray
            The computed divergence as a scalar field with the same shape as the input field.

        Raises
        ------
        ValueError
            If required inputs are missing or invalid.
        """
        # VALIDATE inputs.
        # Coordinates are coerced to CoordinateArray, which ensures a shape at least of (...,NDIM).
        # This does not enforce a grid shape -- must be done when derivatives are assessed.
        coordinates = CoordinateArray(coordinates, self.NDIM)
        # Construct the axes and their indices. Fill in if values are not provided.
        if axes is None:
            axes = self.AXES[:]
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        axes = [self.ensure_axis_string(axis) for axis in axes]
        axes_indices = [self.ensure_axis_numeric(axis) for axis in axes]

        # The field type is determined based on type and then (if numpy array), we coerce the field further to
        # match the shape of the coordinates.
        if isinstance(field, np.ndarray):
            field_type = 'array'

            # checking that the field has the correct shape. This will ensure that
            # (..., NDIM) matches the shape of the coordinates.
            try:
                field = field.reshape((*coordinates.shape[:-1],len(axes)))
            except Exception as e:
                raise ValueError(
                    f"Field (shape={field.shape}) does not match the shape of coordinates (shape={coordinates.shape}).\n"
                    f"The field should have had shape ({coordinates.shape[:-1]},{len(axes)}).\n"
                    f"ERROR={e}")

        elif callable(field):
            field_type = 'function'
        else:
            raise ValueError(
                f"Unsupported field type: {type(field)}. Expected str, sympy.Basic, np.ndarray, or Callable.")

        # MANAGE the derivatives. If the user provided them, we just need to ensure that they are valid. If not, they
        # need to be computed and we need to enforce further restrictions on the grid shape.
        if derivatives is not None:
            if field_type == 'function':
                # validate that the derivatives are provided correctly and that we have as many as
                # necessary.
                if not isinstance(derivatives, (list, tuple)):
                    raise ValueError(f"Field was specified as a function, but derivatives were provided with type {type(derivatives)}.\n"
                                     "We always expect a list of derivative functions when `field` is a function.")

                if len(derivatives) != len(axes):
                    raise ValueError(f"Computing the gradient over {len(axes)} axes, but the user provided {len(derivatives)} derivatives, which doesn't match.")

                # Coerce the coordinates and then evaluate them.
                # The derivatives should now have shape (...,len(axes))
                d_coords = np.moveaxis(coordinates,-1,0)
                derivatives = np.stack([d(*d_coords) for d in derivatives], axis=-1)

            # validate
            if derivatives.shape != (*coordinates.shape[:-1], len(axes)):
                raise ValueError(f"Derivatives were found to have shape {derivatives.shape} but shape should have been {(*coordinates.shape[:-1], len(axes))}... "
                                 f" Please ensure that you have provided the `derivatives` argument correctly. Consult the documentation for detailed information.")

        else:
            # The derivatives are not provided to us; they need to be computed which will require additional enforced
            # structure on the grid.
            if field_type == 'function':
                derivatives = function_partial_derivative(field, coordinates, axes=axes_indices, **kwargs)
            else:
                # validate the grid shape.
                if coordinates.ndim != self.NDIM + 1:
                    raise ValueError(f"Coordinates must be a grid for numerical derivative computations. Expected {self.NDIM +1} dimensional coordinate array.")
                derivatives = partial_derivative(coordinates, field, axes=axes_indices, **kwargs)

        # Compute divergence using the product rule and _get_D_term
        divergence_terms = []
        lame = self.eval_lame(coordinates, axes)
        _scale = dict(unit=-1, contravariant=-2, covariant=0)[basis]
        for i, axis in enumerate(axes):
            # Retrieve or compute the D_term
            D_values = self.D_term(coordinates, axis, basis=basis)
            # Compute the divergence term for this axis
            divergence_term = (D_values * field[..., i]) + (lame[...,i]**_scale)*derivatives[...,i]
            divergence_terms.append(divergence_term)

        # Sum the divergence terms to get the total divergence
        divergence = np.sum(divergence_terms,axis=0)

        return divergence

    def compute_laplacian(self,
                          field: Union[np.ndarray, Callable],
                          coordinates: np.ndarray,
                          /,
                          axes: Union['AxisAlias', List['AxisAlias']] = None,
                          *,
                          basis: str = 'unit',
                          first_derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                          second_derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                          **kwargs) -> np.ndarray:
        r"""
        Compute the Laplacian of a scalar field.

        Parameters
        ----------
        field : Union[np.ndarray, Callable]
            The scalar field to take the gradient of:
            - A numpy array for numerical input. This must be a ``(...,)`` array matching the shape of the
              ``coordinates`` argument (up to the final dimension).
            - A callable function returning a numpy array for functional input. This should have signature
              ``f(x_0,...,x_1)`` and return an array of the same shape as each of the input coordinates.

            .. warning::

                If derivatives are **not** provided explicitly, then fields passed as ``np.ndarray`` must have a more
                stringent grid shape of ``(N_1,N_2,...,N_DIM_FREE,)`` in order to compute the necessary derivatives.

        coordinates : np.ndarray
            Array of coordinates with shape ``(..., NDIM)``, where ``NDIM`` is the number of **free** dimensions. If numerical
            derivatives are necessary, then ``(..., NDIM)`` must be a grid with more stringent shape ``(N_1,...,N_NDIM,N_DIM)``,
            where ``N_i`` may be any number of grid points.

            .. note::

                The free dimensions are those which are free **after** the operation, not before it. Thus,
                if a gradient computation breaks symmetry, the coordinates must include the now-free coordinates.
                
        axes : Union['AxisAlias', List['AxisAlias']], optional
            The axes along which the Laplacian is computed.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            The basis in which to compute the Laplacian. Default is 'unit'.
        first_derivatives : List[Optional[Union[np.ndarray, Callable]]], optional
            Known 1-st derivatives along specific axes to be used instead of numerical differentiation. If the ``field``
            is specified as a ``callable``, then ``derivatives`` should be a ``list[Callable]`` providing each of the
            derivatives to be computed (length of ``axes``). If the field is an array, ``derivatives`` should be an array
            matching the shape of the ``coordinates`` up to the final axis and then having ``len(axes)`` elements in the final
            index.
        second_derivatives : List[Optional[Union[np.ndarray, Callable]]], optional
            Precomputed second derivatives along the specified axes. The structure is similar to
            `first_derivatives`.
        **kwargs
            Additional keyword arguments for numerical differentiation.

        Returns
        -------
        np.ndarray
            The computed Laplacian of the field, with shape ``(...)``.

        Notes
        -----
        The Laplacian of a scalar field :math:`f` in orthogonal coordinates is given by:

        .. math::

            \nabla^2 f = \frac{1}{\lambda_k^2} \partial_k^2 f + \frac{1}{J} \partial_k \left( \frac{J}{\lambda_k^2} \right) \partial_k f.

        This method accounts for contributions from both the second derivative and the first derivative
        scaled by the geometry-dependent :math:`L_k` term. The computation is performed in the specified
        basis, which determines the scaling of the derivative terms.
        """
        # VALIDATE inputs.
        # Coordinates are coerced to CoordinateArray, which ensures a shape at least of (...,NDIM).
        # This does not enforce a grid shape -- must be done when derivatives are assessed.
        coordinates = CoordinateArray(coordinates, self.NDIM)
        # Construct the axes and their indices. Fill in if values are not provided.
        if axes is None:
            axes = self.AXES[:]
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        axes = [self.ensure_axis_string(axis) for axis in axes]
        axes_indices = [self.ensure_axis_numeric(axis) for axis in axes]

        # The field type is determined based on type and then (if numpy array), we coerce the field further to
        # match the shape of the coordinates.
        if isinstance(field, np.ndarray):
            field_type = 'array'

            # checking that the field has the correct shape. This will ensure that
            # (..., NDIM) matches the shape of the coordinates.
            try:
                field = field.reshape((*coordinates.shape[:-1],len(axes)))
            except Exception as e:
                raise ValueError(
                    f"Field (shape={field.shape}) does not match the shape of coordinates (shape={coordinates.shape}).\n"
                    f"The field should have had shape ({coordinates.shape[:-1]},{len(axes)}).\n"
                    f"ERROR={e}")

        elif callable(field):
            field_type = 'function'
        else:
            raise ValueError(
                f"Unsupported field type: {type(field)}. Expected str, sympy.Basic, np.ndarray, or Callable.")

        # MANAGE the derivatives. If the user provided them, we just need to ensure that they are valid. If not, they
        # need to be computed and we need to enforce further restrictions on the grid shape.
        if first_derivatives is not None:
            if field_type == 'function':
                # validate that the first_derivatives are provided correctly and that we have as many as
                # necessary.
                if not isinstance(first_derivatives, (list, tuple)):
                    raise ValueError(f"Field was specified as a function, but first_derivatives were provided with type {type(first_derivatives)}.\n"
                                     "We always expect a list of derivative functions when `field` is a function.")

                if len(first_derivatives) != len(axes):
                    raise ValueError(f"Computing the gradient over {len(axes)} axes, but the user provided {len(first_derivatives)} first_derivatives, which doesn't match.")

                # Coerce the coordinates and then evaluate them.
                # The first_derivatives should now have shape (...,len(axes))
                d_coords = np.moveaxis(coordinates,-1,0)
                first_derivatives = np.stack([d(*d_coords) for d in first_derivatives], axis=-1)

            # validate
            if first_derivatives.shape != (*coordinates.shape[:-1], len(axes)):
                raise ValueError(f"Derivatives were found to have shape {first_derivatives.shape} but shape should have been {(*coordinates.shape[:-1], len(axes))}... "
                                 f" Please ensure that you have provided the `first_derivatives` argument correctly. Consult the documentation for detailed information.")

        else:
            # The first_derivatives are not provided to us; they need to be computed which will require additional enforced
            # structure on the grid.
            if field_type == 'function':
                first_derivatives = function_partial_derivative(field, coordinates, axes=axes_indices, **kwargs)
            else:
                # validate the grid shape.
                if coordinates.ndim != self.NDIM + 1:
                    raise ValueError(f"Coordinates must be a grid for numerical derivative computations. Expected {self.NDIM +1} dimensional coordinate array.")
                first_derivatives = partial_derivative(coordinates, field, axes=axes_indices, **kwargs)

        # MANAGE the derivatives. If the user provided them, we just need to ensure that they are valid. If not, they
        # need to be computed and we need to enforce further restrictions on the grid shape.
        if second_derivatives is not None:
            if field_type == 'function':
                # validate that the second_derivatives are provided correctly and that we have as many as
                # necessary.
                if not isinstance(second_derivatives, (list, tuple)):
                    raise ValueError(f"Field was specified as a function, but second_derivatives were provided with type {type(second_derivatives)}.\n"
                                     "We always expect a list of derivative functions when `field` is a function.")

                if len(second_derivatives) != len(axes):
                    raise ValueError(f"Computing the gradient over {len(axes)} axes, but the user provided {len(second_derivatives)} second_derivatives, which doesn't match.")

                # Coerce the coordinates and then evaluate them.
                # The second_derivatives should now have shape (...,len(axes))
                d_coords = np.moveaxis(coordinates,-1,0)
                second_derivatives = np.stack([d(*d_coords) for d in second_derivatives], axis=-1)

            # validate
            if second_derivatives.shape != (*coordinates.shape[:-1], len(axes)):
                raise ValueError(f"Derivatives were found to have shape {second_derivatives.shape} but shape should have been {(*coordinates.shape[:-1], len(axes))}... "
                                 f" Please ensure that you have provided the `second_derivatives` argument correctly. Consult the documentation for detailed information.")

        else:
            # The second_derivatives are not provided to us; they need to be computed which will require additional enforced
            # structure on the grid.
            # validate the grid shape.
            if coordinates.ndim != self.NDIM + 1:
                raise ValueError(f"Coordinates must be a grid for numerical derivative computations. Expected {self.NDIM +1} dimensional coordinate array.")
            second_derivatives = np.stack([
                partial_derivative(coordinates, first_derivatives[...,axi], axes=[axi], **kwargs) for axi in range(len(axes_indices))],axis=-1)

        # Compute Laplacian terms
        laplacian_terms = []
        lame = self.eval_lame(coordinates, axes)
        for i, axis in enumerate(axes):
            # Compute L term and Lame coefficient
            L_term = self.L_term(coordinates, axis)

            # Combine second derivative and first derivative contributions
            laplacian_term = (1 / lame[...,i] ** 2) * second_derivatives[...,i] + (L_term * first_derivatives[...,i])
            laplacian_terms.append(laplacian_term)

        # Sum the Laplacian terms to get the total Laplacian
        laplacian = np.sum(laplacian_terms, axis=0)

        return laplacian

    def analytical_gradient(self, expression: sp.Basic) -> Dict[str, sp.Basic]:
        """
        Compute the gradient of a scalar expression in the coordinate system.

        Parameters
        ----------
        expression : sp.Basic
            A symbolic expression representing the scalar field.

        Returns
        -------
        Dict[str, sp.Basic]
            A dictionary mapping each axis to its gradient component.
        """
        gradient = {}
        for axis in self.AXES:
            lame_coeff = self.get_lame_symbolic(axis)
            gradient[axis] = sp.simplify(sp.diff(expression, self.SYMBAXES[self.AXES.index(axis)]) / lame_coeff)
        return gradient

    def analytical_divergence(self, vector_field: Dict[str, sp.Basic]) -> sp.Basic:
        """
        Compute the divergence of a vector field in the coordinate system.

        Parameters
        ----------
        vector_field : Dict[str, sp.Basic]
            A dictionary mapping each axis to a symbolic expression representing the vector field component.

        Returns
        -------
        sp.Basic
            A symbolic expression for the divergence of the vector field.
        """
        divergence = 0
        jacobian = self.get_derived_attribute_symbolic('jacobian')
        for axis in self.AXES:
            lame_coeff = self.get_lame_symbolic(axis)
            term = sp.diff(jacobian * vector_field[axis] / lame_coeff, self.SYMBAXES[self.AXES.index(axis)])
            divergence += term / jacobian
        return sp.simplify(divergence)

    def analytical_laplacian(self, expression: sp.Basic) -> sp.Basic:
        """
        Compute the Laplacian of a scalar expression in the coordinate system.

        Parameters
        ----------
        expression : sp.Basic
            A symbolic expression representing the scalar field.

        Returns
        -------
        sp.Basic
            A symbolic expression for the Laplacian of the scalar field.
        """
        laplacian = 0
        jacobian = self.get_derived_attribute_symbolic('jacobian')
        for axis in self.AXES:
            # Compute first set of values
            lame_coeff = self.get_lame_symbolic(axis)
            first_deriv = sp.diff(expression, self.SYMBAXES[self.AXES.index(axis)])
            # Compute the second set of values
            _exp = (jacobian/lame_coeff**2)*first_deriv
            second_deriv = sp.diff(_exp, self.SYMBAXES[self.AXES.index(axis)])

            laplacian += (1/jacobian)*second_deriv

        return sp.simplify(laplacian)

    # @@ LAME FUNCTION MANAGEMENT @@ #
    # This is where the developer should implement the relevant Lame coefficient functions and
    # the various utilities for fetching those functions are placed.
    def get_lame_symbolic(self, axis: 'AxisAlias') -> sp.Basic:
        return self._lame_inst_symbols[self.ensure_axis_string(axis)]

    def get_lame_function(self, axis: 'AxisAlias') -> Callable:
        return self._lame_functions[self.ensure_axis_numeric(axis)]

    def eval_lame(self, coordinates, axes: Optional[List['AxisAlias'] | 'AxisAlias'] = None) -> np.ndarray:
        # COERCE coordinates
        coordinates = CoordinateArray(coordinates, self.NDIM)
        c_shape = coordinates.shape
        coordinates = np.moveaxis(coordinates, -1, 0)

        # VALIDATE axes
        if axes is None:
            axes = self.AXES
        if any(ax not in self.AXES for ax in axes):
            raise ValueError("Axes not in AXES list")

        output_array = np.zeros((*c_shape[:-1], len(axes)))

        for axi, ax in enumerate(axes):
            output_array[..., axi] = self.get_lame_function(ax)(*coordinates)

        return output_array

    def scale_by_lame(self, coordinates, field, axes=None, order=1):
        """
        Scale a scalar or vector field by the Lame coefficients of the coordinate system.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of coordinates with shape (..., NDIM).
        field : np.ndarray
            The field to scale. Can be:
            - A scalar field with shape (...), matching the spatial grid of `coordinates`.
            - A vector field with shape (..., N), where `N` corresponds to the number of axes.
        axes : List[str] or None, optional
            The axes along which the field components are defined. If not provided, it is deduced
            from the last dimension of `field`.
        order : int, optional
            The order of scaling by Lame coefficients. Default is 1.

        Returns
        -------
        np.ndarray
            The scaled field with the same shape as the input `field`.

        Raises
        ------
        ValueError
            If the field dimensions do not match the coordinate or axis specifications.
        """
        # Coerce coordinates to the correct format
        coordinates = CoordinateArray(coordinates, self.NDIM)

        # Extract the grid shape from coordinates
        grid_shape = coordinates.shape[:-1]

        # Determine if the field is scalar or vector and adjust its shape
        _is_scalar = False
        if field.shape == grid_shape:  # Scalar field
            _is_scalar = True
            field = field[..., np.newaxis]  # Add a dummy axis for uniform processing

        # Validate axes and match with field components
        if axes is None:
            axes = self.AXES[:field.shape[-1]]  # Deduce axes from the field's last dimension
        if len(axes) != field.shape[-1]:
            raise ValueError("`axes` must have the same length as the field's last dimension.")

        # Evaluate Lame coefficients for the specified axes
        lame_coefficients = self.eval_lame(coordinates, axes=axes)

        # Scale the field by the Lame coefficients raised to the specified order
        scaled_field = field * (lame_coefficients ** order)

        # If the input field was scalar, return a scalar result
        if _is_scalar:
            return scaled_field.squeeze(axis=-1)

        return scaled_field


    # @@ COORDINATE CONVERSION @@ #
    # These are handlers for converting between coordinate systems.
    # The _convert_native_to_cartesian and _convert_cartesian_to_native methods
    # should be implemented uniquely for each subclass and the rest is performed automatically.
    def to_cartesian(self, coordinates) -> np.ndarray:
        """
        Convert native coordinates of this coordinate system to Cartesian coordinates.

        This method transforms coordinates from the system defined by this instance to the standard
        Cartesian system (e.g., [x, y, z] in 3D space). The conversion function ``_convert_native_to_cartesian``
        must be implemented by subclasses for this operation.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of native coordinates in this coordinate system, with shape ``(..., NDIM)``, where ``...`` is
            the grid of points and ``NDIM`` is the number of dimensions.

        Returns
        -------
        np.ndarray
            Array of Cartesian coordinates with shape ``(...,NDIM)``.

        Raises
        ------
        ConversionError
            If the conversion fails, this error provides a message indicating the failure cause.

        Notes
        -----
        The conversion relies on the specific coordinate transformation implemented in each subclass.
        This function is typically useful in applications requiring alignment with the Cartesian
        coordinate system for further operations or visualizations.

        See Also
        --------
        from_cartesian : Converts Cartesian coordinates to this native coordinate system.
        """
        # COERCE the coordinates
        coordinates = CoordinateArray(coordinates)

        # PERFORM the conversion
        try:
            return self._convert_native_to_cartesian(coordinates)
        except Exception as e:
            raise ConversionError(f"Failed to convert from {self.__class__.__name__} to cartesian: {e}")

    def from_cartesian(self, coordinates) -> np.ndarray:
        """
        Convert Cartesian coordinates to the native coordinates of this coordinate system.

        This method converts points from the Cartesian coordinate system into the coordinates
        specific to this coordinate system instance. The conversion function ``_convert_cartesian_to_native``
        must be implemented by subclasses for this operation.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of Cartesian coordinates with shape ``(..., NDIM)``, where ``...`` may be any grid structure and
            ``NDIM`` is the number of dimensions.

        Returns
        -------
        np.ndarray
            Array of coordinates in this coordinate system with shape ``(..., NDIM)``, where ``NDIM``
            is the number of dimensions of this coordinate system.

        Raises
        ------
        ConversionError
            If the conversion fails, this error provides a message indicating the failure cause.

        Notes
        -----
        Conversion between coordinate systems can be useful for transformations, analysis, or plotting.
        The specific transformation depends on each subclass’s geometry and is implemented in
        ``_convert_cartesian_to_native``.

        See Also
        --------
        to_cartesian : Converts native coordinates in this system to Cartesian coordinates.
        """
        # COERCE the coordinates
        coordinates = CoordinateArray(coordinates)

        # PERFORM the conversion
        try:
            return self._convert_cartesian_to_native(coordinates)
        except Exception as e:
            raise ConversionError(f"Failed to convert from cartesian to  {self.__class__.__name__}: {e}")

    def convert_to(self, target_coord_system: 'CoordinateSystem', *args: Any) -> np.ndarray:
        """
        Convert coordinates from this system to another specified coordinate system.

        This method provides a general mechanism for transforming coordinates between any two
        orthogonal coordinate systems by first transforming from the native system to Cartesian
        coordinates, then to the target coordinate system. The conversion is handled by a
        ``CoordinateConverter`` instance.

        Parameters
        ----------
        target_coord_system : CoordinateSystem
            The coordinate system to which coordinates will be converted.
        *args : Any
            Coordinates in this system's native format to be converted.

        Returns
        -------
        np.ndarray
            Array of coordinates in the target coordinate system's format.

        Notes
        -----
        This method is a higher-level interface for coordinate transformation between orthogonal
        systems. It relies on the Cartesian system as an intermediary step to standardize the
        transformation.

        Raises
        ------
        ConversionError
            If any part of the conversion process fails, including intermediate transformations.

        See Also
        --------
        to_cartesian : Converts coordinates in this system to Cartesian coordinates.
        from_cartesian : Converts Cartesian coordinates to this coordinate system.
        """
        converter = CoordinateConverter(self, target_coord_system)
        return converter(*args)

    @abstractmethod
    def _convert_native_to_cartesian(self, coordinates: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _convert_cartesian_to_native(self, coordinates: np.ndarray) -> np.ndarray:
        pass

    # @@ UTILITY FUNCTIONS @@ #
    # These are basic helper functions to be used in various other methods.
    def ensure_axis_numeric(self, axis: 'AxisAlias') -> int:
        if isinstance(axis, str):
            return self.AXES.index(axis)
        else:
            return axis

    def ensure_axis_string(self, axis: 'AxisAlias') -> str:
        if isinstance(axis, int):
            return self.AXES[axis]
        else:
            return axis

    def ensure_axis_order(self, axes):
        axes = [self.ensure_axis_string(ax) for ax in axes]
        return [ax for ax in self.AXES if ax in axes]

    # @@ IO OPERATIONS @@ #
    # These provide method for reading and writing coordinate systems to disk. They
    # generally do not need to be overwritten in custom implementations.
    def to_file(self, file_obj, fmt: str = 'json'):
        """
        Save the coordinate system configuration to a file or group.

        Parameters
        ----------
        file_obj : file-like object
            The open file or group to save to.
        fmt : {'json', 'yaml', 'hdf5'}, optional
            The format to use for saving. Default is 'json'.
        """
        if fmt == 'json':
            self._to_json(file_obj)
        elif fmt == 'yaml':
            self._to_yaml(file_obj)
        elif fmt == 'hdf5':
            self._to_hdf5(file_obj)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json', 'yaml', or 'hdf5'.")

    @classmethod
    def from_file(cls, file_obj, fmt: str = 'json'):
        """
        Load the coordinate system configuration from a file or group.

        Parameters
        ----------
        file_obj : file-like object
            The open file or group to load from.
        fmt : {'json', 'yaml', 'hdf5'}, optional
            The format to use for loading. Default is 'json'.

        Returns
        -------
        CoordinateSystem
            An instance of the loaded coordinate system.
        """
        if fmt == 'json':
            return cls._from_json(file_obj)
        elif fmt == 'yaml':
            return cls._from_yaml(file_obj)
        elif fmt == 'hdf5':
            return cls._from_hdf5(file_obj)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'json', 'yaml', or 'hdf5'.")

    def _to_json(self, file_obj):
        """
        Save configuration to JSON format.

        Parameters
        ----------
        file_obj : file-like object
            An open file to write to.
        """
        import json
        data = {
            'class_name': self.__class__.__name__,
            'parameters': self.parameters,
        }
        json.dump(data, file_obj)

    @classmethod
    def _from_json(cls, file_obj):
        """
        Load configuration from JSON format.

        Parameters
        ----------
        file_obj : file-like object
            An open file to read from.

        Returns
        -------
        CoordinateSystem
            An instance of the loaded coordinate system.
        """
        import json
        data = json.load(file_obj)

        _cls = find_in_subclasses(CoordinateSystem, data['class_name'])
        return _cls(**data['kwargs'])

    def _to_yaml(self, file_obj):
        """
        Save configuration to YAML format.

        Parameters
        ----------
        file_obj : file-like object
            An open file to write to.
        """
        import yaml
        data = {
            'class_name': self.__class__.__name__,
            'parameters': self.parameters,
        }
        yaml.dump(data, file_obj)

    @classmethod
    def _from_yaml(cls, file_obj):
        """
        Load configuration from YAML format.

        Parameters
        ----------
        file_obj : file-like object
            An open file to read from.

        Returns
        -------
        CoordinateSystem
            An instance of the loaded coordinate system.
        """
        import yaml
        data = yaml.safe_load(file_obj)

        _cls = find_in_subclasses(CoordinateSystem, data['class_name'])
        return _cls(**data['kwargs'])

    def _to_hdf5(self, group_obj):
        """
        Save configuration to HDF5 format.

        Parameters
        ----------
        group_obj : h5py.Group
            An open HDF5 group to write to.
        """
        import json
        group_obj.attrs['class_name'] = self.__class__.__name__

        # Save each kwarg individually as an attribute
        for key, value in self.parameters.items():
            if isinstance(value, (int, float, str)):
                group_obj.attrs[key] = value
            else:
                group_obj.attrs[key] = json.dumps(value)  # serialize complex data

    @classmethod
    def _from_hdf5(cls, group_obj):
        """
        Load configuration from HDF5 format.

        Parameters
        ----------
        group_obj : h5py.Group
            An open HDF5 group to read from.

        Returns
        -------
        CoordinateSystem
            An instance of the loaded coordinate system.
        """
        import json
        data = {
            'class_name': group_obj.attrs['class_name'],
        }

        # Load kwargs, deserializing complex data as needed
        kwargs = {}
        for key, value in group_obj.attrs.items():
            if key != 'class_name':
                try:
                    kwargs[key] = json.loads(value)  # try to parse complex JSON data
                except (TypeError, json.JSONDecodeError):
                    kwargs[key] = value  # simple data types remain as is

        data['kwargs'] = kwargs

        _cls = find_in_subclasses(CoordinateSystem, data['class_name'])
        return _cls(**data['kwargs'])

    def __hash__(self):
        """
        Compute a hash value for the CoordinateSystem instance.

        The hash is based on the class name, positional arguments (`_args`), and keyword arguments (`_kwargs`).
        This ensures that two instances with the same class and initialization parameters produce the same hash.

        Returns
        -------
        int
            The hash value of the instance.
        """
        return hash((
            self.__class__.__name__,
            tuple(sorted(self.parameters.items()))
        ))

class RadialCoordinateSystem(CoordinateSystem, ABC):
    """
    Base class for radially defined coordinate systems.

    A coordinate system is **Radial** if its first axis has level surfaces which form shells around
    the origin. In this case, several additional operations are made considerably simpler.
    """
    # @@ CLASS ATTRIBUTES @@ #
    # DEVELOPERS: these should be set in any subclass of CoordinateSystem. The axes
    #  should follow standard convention and the parameters and axes will become the
    #  symbolic attributes of the class.
    NDIM: int = 3
    AXES: list[str] = ['x','y','z']
    PARAMETERS: Dict[str, Any] = {}

    # @@ DYNAMICALLY GENERATED ATTRIBUTES @@ #
    # These attributes are generated dynamically in the metaclass
    # and then populated based on other class methods.
    #
    # DEVELOPERS: If changes are made to the metaclass, new variables could be added here
    #  to ensure that IntelliJ understands what's going on.
    SYMBAXES: List[sp.Symbol] = None
    SYMBPARAMS: Dict[str, sp.Symbol] = None
    _lame_symbolic: Dict[str, sp.Basic] = None

    @abstractmethod
    def integrate_in_shells(self,
                            field: Union[np.ndarray,Callable],
                            radii: np.ndarray):
        pass

    @abstractmethod
    def integrate_to_infinity(self,
                              field: Union[np.ndarray,Callable],
                              radii: np.ndarray):
        pass