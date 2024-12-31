"""
Utilities for computations under symmetry constraints.

This module provides utilities for managing geometrical operations, symmetry,
and coordinate systems in multi-dimensional spaces. It facilitates handling
of complex geometrical structures, transformations, and operations such as
gradients, divergences, and Laplacians in a consistent and efficient manner.

The core focus is on enabling seamless integration of coordinate systems with
invariance properties, often encountered in physical simulations and mathematical
modeling. Symmetry management is a key feature, allowing users to optimize
computations by leveraging axis invariance.
"""
from typing import List, Optional, Union, Callable, TYPE_CHECKING

import numpy as np

from pisces.geometry.base import CoordinateSystem
from pisces.utilities.array_utils import CoordinateArray, fill_missing_coord_axes, CoordinateGrid
from pisces.utilities.general import find_in_subclasses
from pisces.utilities.logging import devlog

if TYPE_CHECKING:
    from pisces.geometry._typing import AxisAlias

class GeometryHandler:
    """
    A utility class for managing geometry operations under symmetry constraints.

    The :py:class:`GeometryHandler` provides methods and properties to handle computations in
    multi-dimensional coordinate systems while accounting for symmetry. It allows
    the definition of free and symmetric axes, manages coordinate transformations,
    and facilitates operations such as gradients, divergences, and Laplacians.

    The class is designed to be flexible and extendable, enabling specialized handlers
    for specific coordinate systems by dynamically instantiating the appropriate subclass.

    Features
    --------

    - **Symmetry Management**: Differentiates between free and symmetric axes, allowing
      optimized computations by ignoring symmetric dimensions.
    - **Coordinate Handling**: Provides methods to coerce partial coordinates into full
      coordinates based on symmetry, ensuring compatibility with the coordinate system.
    - **Dependence Analysis**: Identifies dependencies for gradient, divergence, and
      Laplacian operations, ensuring only necessary computations are performed.
    - **Dynamic Subclassing**: Automatically selects the appropriate subclass based on
      the coordinate system's `_handler_class_name` attribute.

    Notes
    -----
    - This class serves as a base class for specialized geometry handlers that may
      implement additional features or optimizations for specific coordinate systems.
    - The ``_handler_class_name`` attribute in the associated coordinate system determines
      the appropriate subclass to instantiate.
    """
    def __new__(cls,
                coordinate_system: CoordinateSystem,
                /,
                free_axes: Optional[List[str]] = None,
                *args,
                fill_values: Optional[float | List[float]] = 0.0,
                **kwargs):
        """
        Dynamically create a :py:class:`GeometryHandler` instance based on the provided coordinate system.

        This method identifies the appropriate subclass of :py:class:`GeometryHandler` to handle the
        given ``coordinate_system``. It looks for a subclass registered to the coordinate system's
        handler name and returns an instance of that subclass. If no specific handler subclass
        is found, it defaults to the base :py:class:`GeometryHandler` class.

        Parameters
        ----------
        coordinate_system : :py:class:`CoordinateSystem`
            The coordinate system for which a geometry handler is being instantiated.
            The coordinate system class must define a ``_handler_class_name`` attribute to indicate
            the appropriate handler class.
        free_axes : Optional[List[str]], optional
            A list of axes that are not affected by symmetry. If not specified, all axes in the
            coordinate system are considered free.
        *args : tuple
            Positional arguments passed to the :py:class:`GeometryHandler` or its subclass.
        fill_values : Optional[float | List[float]], optional
            Default fill values for the symmetric (fixed) axes. These values are used to populate
            missing dimensions in the coordinate system.
        **kwargs : dict
            Additional keyword arguments passed to the :py:class:`GeometryHandler` or its subclass.

        Returns
        -------
        :py:class:`GeometryHandler`
            An instance of the appropriate subclass of :py:class:`GeometryHandler`.

        Raises
        ------
        ValueError
            If the handler class name specified in ``_handler_class_name`` does not match any
            subclasses of :py:class:`GeometryHandler`.

        Notes
        -----
        - This method assumes that each :py:class:`CoordinateSystem` subclass specifies a `_handler_class_name`
          attribute, which is a string matching the name of its corresponding :py:class:`GeometryHandler` subclass.
        - If the class name matches the base :py:class:`GeometryHandler`, no subclass lookup is performed,
          and the base class is instantiated directly.
        """
        # Retrieve the handler class name from the coordinate system that was passed. If we cannot
        # find one, then we assume the :py:class:`GeometryHandler` base class.
        if hasattr(coordinate_system, '_handler_class_name'):
            # We have a handler class name specifier -> look up the class.
            handler_class_name = getattr(coordinate_system,'_handler_class_name')
            if handler_class_name == cls.__name__:
                handler_subclass = cls
            else:
                handler_subclass = find_in_subclasses(cls, handler_class_name)
        else:
            devlog.warning("Failed to find a _handler_class_name attribute on CoordinateSystem subclass %s.",
                           coordinate_system.__class__.__name__)
            handler_subclass = cls

        # Return the object in the class specified by the handler subclass attribute.
        return object.__new__(handler_subclass)


    def __init__(self, coordinate_system: CoordinateSystem,
                 /,
                 free_axes: Optional[List[str]] = None,
                 *_,
                 fill_values: Optional[float|List[float]] = 0.0,
                 **__):
        """
        Initialize the :py:class:`GeometryHandler` with the given coordinate system and symmetry constraints.

        Parameters
        ----------
        coordinate_system : :py:class:`CoordinateSystem`
            The coordinate system for which the geometry handler is being initialized.

            .. note::

                In some cases, specific :py:class:`CoordinateSystem`s will redirect :py:class:`GeometryHandler` to
                actually initialize a subclass of :py:class:`GeometryHandler`. These specialized handlers are used to
                manage special properties of these specific :py:class:`CoordinateSystem` classes.

        free_axes : Optional[List[str]], optional
            A list of the axes which are **NOT** constrained by symmetry.
            These axes should be the coordinates that are still relevant once symmetry has been accounted for.

            .. hint::

                If a function :math:`f(x,y,z)` is invariant under changes in :math:`z`, but changes with the other
                two axes, then ``free_axes = ['x','y']``.

        *args :
            Additional positional arguments. These are not used by the base class.
        fill_values : Optional[float | List[float]], optional
            Default fill values for the symmetric (fixed) axes. These values are used
            to populate missing dimensions in the coordinate system. Default is ``0.0``. If a ``float`` is provided,
            it will be applied to all the symmetric axes. Otherwise, it must be ``NDIM_SYM`` in shape specifying
            the fill have for each of the symmetric axes.
        **kwargs :
            Additional keyword arguments. These are not used by the base class.

        Raises
        ------
        ValueError
            If ``free_axes`` contains axes not defined in the coordinate system.
            If the length of ``fill_values`` does not match the number of symmetric axes.

        Notes
        -----
        - The ``free_axes`` are the axes not affected by symmetry and are preserved
          for operations. The remaining axes are treated as symmetric and assigned
          the specified ``fill_values``.
        - Cached masks (``_symmetric_mask`` and ``_free_mask``) are initialized for
          efficient processing of free and symmetric axes during computations.
        """
        # CONFIGURING the core attributes.
        # Set the coordinate system, identify the relevant symmetry axes and then
        # proceed with initialization.
        self.coordinate_system: CoordinateSystem = coordinate_system
        """:py:class:`CoordinateSystem`: The coordinate system of the geometry handler."""

        # Manage the free axes.
        if free_axes is None:
            free_axes = self.coordinate_system.AXES[:]

        if any(ax not in self.coordinate_system.AXES for ax in free_axes):
            raise ValueError("Coordinate axes are not defined")

        # Enforce ordering.
        self._free_axes = [ax for ax in self.coordinate_system.AXES if ax in free_axes]
        self._sym_axes = [ax for ax in self.coordinate_system.AXES if ax not in free_axes]

        # Manage the fill values
        if isinstance(fill_values, float):
            fill_values = np.array(len(self._sym_axes)*[fill_values])

        if len(fill_values) != len(self._sym_axes):
            raise ValueError(f"fill_values must have the same length as {len(self._sym_axes)}")

        self.fill_values = fill_values
        """ndarray: The fill values for each of the symmetric axes."""

        # CREATING cached attributes for storing generated attributes
        # during the lifecycle of this class.
        self._symmetric_mask = None
        self._free_mask = None

    def get_fill_values(self, fixed_axes: Optional[Union[List[str], str]] = None) -> np.ndarray:
        """
        Fetch the fill values corresponding to the specified fixed (symmetric) axes.

        Parameters
        ----------
        fixed_axes : Optional[Union[List[str], str]], optional
            A subset of :py:attr:`GeometryHandler.fixed_axes` for which to obtain the fill values.

        Returns
        -------
        np.ndarray
            An array of fill values corresponding to the specified axes.

        Raises
        ------
        ValueError
            If any axis in ``fixed_axes`` is not in :py:attr:`GeometryHandler.fixed_axes`.
        """
        # Ensure `fixed_axes` is a list
        if isinstance(fixed_axes, str):
            fixed_axes = [fixed_axes]
        elif fixed_axes is None:
            fixed_axes = self._sym_axes  # Default to all symmetric axes

        # Validate that the specified axes are a subset of symmetric axes
        invalid_axes = [ax for ax in fixed_axes if ax not in self._sym_axes]
        if invalid_axes:
            raise ValueError(f"Invalid axes {invalid_axes}. Must be a subset of symmetric axes {self._sym_axes}.")

        # Fetch indices and corresponding fill values
        fixed_indices = np.array([self._sym_axes.index(ax) for ax in fixed_axes])
        return np.array(self.fill_values)[fixed_indices]

    @classmethod
    def coerce_coordinates(cls,
                           coordinates: np.ndarray,
                           coordinate_system: CoordinateSystem,
                           free_axes: List[str],
                           fill_values: float | List[float] = 0.0) -> np.ndarray:
        """
        Coerce a set of partial coordinates into full coordinates based on the specified free axes and fill values.

        Parameters
        ----------
        coordinates : np.ndarray
            The partial coordinates array with shape ``(..., NDIM_FREE)``, where ``NDIM_FREE ``is the number of free axes.
        coordinate_system : :py:class:`CoordinateSystem`
            The coordinate system to use for coercion, defining the full set of axes.
        free_axes : List[str]
            The subset of axes considered free, corresponding to the provided ``coordinates``.
        fill_values : float or List[float], optional
            The fill values for the symmetric (non-free) axes. Default is ``0.0``.

        Returns
        -------
        np.ndarray
            The full coordinates array with shape ``(..., NDIM)``, where ``NDIM`` is the number of dimensions in the coordinate system.

        Raises
        ------
        ValueError
            If the free axes are not valid, or if the shape of ``coordinates`` or ``fill_values` `does not match expectations.
        """
        # Validate: look at the free_axes, ensure that we have those axes and generate a mask from them.
        if any(ax not in coordinate_system.AXES for ax in free_axes):
            raise ValueError(f"Coordinates specified in `free_axes` were {free_axes}, which was not a subset of coordinate system"
                             f" axes {coordinate_system.AXES}.")

        free_axes_mask = np.array([ax in free_axes for ax in coordinate_system.AXES],dtype=bool)
        fixed_axes_mask = ~free_axes_mask

        # Manage the fill values
        fill_values = np.array(fill_values)
        if fill_values.ndim == 0:
            fill_values = np.tile(fill_values, np.sum(fixed_axes_mask))

        try:
            fill_values.reshape((np.sum(fixed_axes_mask),))
        except ValueError as e:
            raise ValueError(f"`fill_values` was length {len(fill_values)}, expected {np.sum(fixed_axes_mask)}.") from e

        # Ensure coordinates is a correctly formatted coordinate set.
        coordinates = CoordinateArray(coordinates, np.sum(free_axes_mask))
        return fill_missing_coord_axes(coordinates, axis_mask=free_axes_mask,fill_values=fill_values)

    @classmethod
    def coerce_coordinate_grid(cls,
                               coordinates: np.ndarray,
                               coordinate_system: CoordinateSystem,
                               free_axes: List[str],
                               fill_values: float | List[float] = 0.0) -> np.ndarray:
        """
        Coerce a set of coordinates into a valid coordinate grid.

        Ensures that the provided coordinates conform to the full dimensions of the
        coordinate system (``NDIM``) by filling in symmetric axes and reshaping to
        represent a structured grid.

        Parameters
        ----------
        coordinates : np.ndarray
            The input coordinates to coerce. Should have shape ``(..., NDIM_FREE)``, where ``NDIM_FREE``
            is the number of free axes. If provided in grid form, additional constraints apply.
        coordinate_system : CoordinateSystem
            The coordinate system defining the full set of axes. Used to determine the complete
            structure of the grid.
        free_axes : List[str]
            The subset of axes that are considered free (not affected by symmetry). These
            correspond to the dimensions of the input coordinates.
        fill_values : float or List[float], optional
            Values to use for filling the symmetric (fixed) axes. If a single float is provided,
            the same value is used for all fixed axes. If a list is provided, it must have a length
            equal to the number of fixed axes. Default is 0.0.

        Returns
        -------
        np.ndarray
            The coerced coordinates as a structured grid with shape ``(..., NDIM)``, where ``NDIM``
            is the total number of dimensions in the coordinate system.

        Raises
        ------
        ValueError
            If the free axes are not valid, or if the shape of ``coordinates`` or ``fill_values`` does
            not match the expectations of the coordinate system.

        Notes
        -----
        - This method first calls `coerce_coordinates` to ensure that the input coordinates are valid
          for the full dimensionality of the coordinate system. It then reshapes the coordinates into
          a structured grid format.
        - The `free_axes` determine the structure of the input coordinates, while the symmetric axes
          are automatically filled using the specified ``fill_values``.

        Examples
        --------
        Let's consider the case where we have a radial function on a 1-D grid representing radii in spherical coordinates.
        We have 100 points in the radius grid. We need to create a fully compliant ``(100,1,1,3)`` grid with the correct
        fill values for the missing coordinates.

        >>> from pisces.geometry import SphericalCoordinateSystem
        >>> coordinate_sys = SphericalCoordinateSystem()
        >>> radii = np.linspace(0,1,100)
        >>> fill_vals = [np.pi/2,0]
        >>> coords = GeometryHandler.coerce_coordinate_grid(radii,coordinate_system=coordinate_sys,free_axes=['r'],fill_values=fill_vals)
        >>> coords.shape
        (100, 1, 1, 3)
        """
        # Coerce the coordinates into a valid format. This will ensure they are ``(...,NDIM)``.
        free_axes_mask = np.array([ax in free_axes for ax in coordinate_system.AXES],dtype=bool)
        coordinates = cls.coerce_coordinates(coordinates, coordinate_system, free_axes, fill_values)

        # Proceed to treat them as a grid. The axes present should be just the free_axes mask.
        return CoordinateGrid(coordinates, ndim=coordinate_system.NDIM, axes_mask=free_axes_mask)

    @classmethod
    def coerce_function(cls,
                        function: Callable,
                        coordinate_system: CoordinateSystem,
                        free_axes: List[str]) -> Callable:
        """
        Wrap a function to operate only on the specified free axes of a coordinate system.

        Parameters
        ----------
        function : Callable
            The original function to be wrapped. It should accept arguments corresponding to
            the free axes of the coordinate system.
        coordinate_system : :py:class:`CoordinateSystem`
            The coordinate system defining the full set of axes.
        free_axes : List[str]
            The subset of axes corresponding to the input arguments of the function.

        Returns
        -------
        Callable
            A wrapped function that operates only on the free axes.

        Raises
        ------
        ValueError
            If any of the free axes are not part of the coordinate system's axes.

        Notes
        -----
        This method ensures that the wrapped function only processes the relevant axes
        and ignores others, making it compatible with the coordinate system's structure.
        """
        # Validate that the free axes are part of the coordinate system's axes
        if any(ax not in coordinate_system.AXES for ax in free_axes):
            raise ValueError(
                f"Axes in {free_axes} are not defined in the coordinate system (axes={coordinate_system.AXES})."
            )

        # Create a mask to extract arguments corresponding to the free axes
        mask = np.array([ax in free_axes for ax in coordinate_system.AXES], dtype=bool)
        axes = coordinate_system.AXES[:]

        def _wrapped_function(*args):
            # Ensure arguments match the number of full axes
            if len(args) != len(axes):
                raise ValueError(
                    f"Expected {len(axes)} arguments, but got {len(args)}."
                )

            # Extract only the arguments corresponding to the free axes
            filtered_args = np.array(args)[mask]
            return function(*filtered_args)

        return _wrapped_function

    def get_gradient_dependence(self,
                                axes: Optional[Union[List[str], str]] = None,
                                basis: str = 'unit') -> List[str]:
        """
        Determine the symbols on which the gradient computation depends for the specified axes.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the gradient computation. If None, defaults to :py:attr:`GeometryHandler.free_axes`.
        basis : str, optional
            The basis in which the divergence is computed. This can be one of the following:

            - 'unit' (default): The unit basis.
            - 'contravariant': The contravariant basis.
            - 'covariant': The covariant basis.

        Returns
        -------
        List[str]
            A list of strings representing the symbols on which the gradient depends.

        Notes
        -----
        This method inspects the symbolic Lame coefficients of the coordinate system to determine the
        dependencies for the specified axes.
        """
        # Default to all free axes if none are specified
        if axes is None:
            axes = self._free_axes
        elif isinstance(axes, str):
            axes = [axes]

        # Validate and filter axes to ensure they are in the free axes
        axes = [ax for ax in axes if ax in self._free_axes]

        # Collect dependencies from the Lame coefficients
        dependent_symbols = set(np.array(self.coordinate_system.SYMBAXES)[self.free_mask])

        if basis != 'contravariant':
            for ax in axes:
                dependent_symbols.update(self.coordinate_system.get_lame_symbolic(ax).free_symbols)

        # Return the symbols as a list of strings
        return self.coordinate_system.ensure_axis_order([str(sym) for sym in dependent_symbols])

    def get_divergence_dependence(self,
                                  axes: Optional[Union[List[str], str]] = None,
                                  basis: str = 'unit') -> List[str]:
        """
        Determine the symbols on which the divergence computation depends.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the divergence computation. If None, defaults to all free axes.
        basis : str, optional
            The basis in which the divergence is computed. This can be one of the following:

            - 'unit' (default): The unit basis.
            - 'contravariant': The contravariant basis.
            - 'covariant': The covariant basis.

        Returns
        -------
        List[str]
            A list of strings representing the symbols on which the divergence depends.

        Notes
        -----
        This method inspects the symbolic D-terms and Lame coefficients of the coordinate system
        to determine the dependencies for the specified axes. The scaling applied to the Lame
        coefficients depends on the chosen basis:

        - 'unit': Scaling factor of -1.
        - 'contravariant': Scaling factor of -2.
        - 'covariant': Scaling factor of 0.

        Dependencies include both the symbolic D-term and the scaled Lame coefficients.
        """
        # Default to all free axes if none are specified
        if axes is None:
            axes = self._free_axes
        elif isinstance(axes, str):
            axes = [axes]

        # Validate and filter axes to ensure they are in the free axes
        axes = [ax for ax in axes if ax in self._free_axes]

        # Collect the dependencies
        dependent_symbols = set(np.array(self.coordinate_system.SYMBAXES)[self.free_mask])

        _scale = dict(unit=-1, contravariant=-2, covariant=0)[basis]
        for ax in axes:
            d_term = self.coordinate_system.get_symbolic_D_term(ax,basis=basis)
            lame_term = self.coordinate_system.get_lame_symbolic(ax)**_scale

            s1 = d_term.free_symbols
            s2 = lame_term.free_symbols

            dependent_symbols.update(s1)
            dependent_symbols.update(s2)

        return self.coordinate_system.ensure_axis_order([str(sym) for sym in dependent_symbols])

    def get_laplacian_dependence(self,
                                 axes: Optional[Union[List[str], str]] = None) -> List[str]:
        """
        Determine the symbols on which the Laplacian computation depends.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the Laplacian computation. If None, defaults to all free axes.

        Returns
        -------
        List[str]
            A list of strings representing the symbols on which the Laplacian depends.

        Notes
        -----
        This method inspects the symbolic L-terms and Lame coefficients of the coordinate system
        to determine the dependencies for the specified axes. Dependencies include:

        - The symbolic L-term, which represents contributions from coordinate geometry.
        - The scaled Lame coefficients, raised to the power of -2, to account for geometric scaling
          in orthogonal coordinate systems.
        """
        # Default to all free axes if none are specified
        if axes is None:
            axes = self._free_axes
        elif isinstance(axes, str):
            axes = [axes]

        # Validate and filter axes to ensure they are in the free axes
        axes = [ax for ax in axes if ax in self._free_axes]

        # Collect the dependencies
        dependent_symbols = set(np.array(self.coordinate_system.SYMBAXES)[self.free_mask])

        for ax in axes:
            L_term = self.coordinate_system.get_symbolic_L_term(ax)
            lame_term = self.coordinate_system.get_lame_symbolic(ax) ** (-2)

            s1 = L_term.free_symbols
            s2 = lame_term.free_symbols

            dependent_symbols.update(s1)
            dependent_symbols.update(s2)

        return self.coordinate_system.ensure_axis_order([str(sym) for sym in dependent_symbols])

    def get_gradient_descendant(self,
                                axes: Optional[Union[List[str], str]] = None,
                                basis: str = 'unit') -> 'GeometryHandler':
        """
        Create a descendant:py:class:`GeometryHandler` instance that includes only the axes necessary for gradient computations.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the gradient computation. If None, defaults to all free axes.
        basis : str, optional
            The basis in which the gradient is computed. This can be one of the following:

            - 'unit' (default): The unit basis.
            - 'contravariant': The contravariant basis.
            - 'covariant': The covariant basis.

        Returns
        -------
        GeometryHandler
            A new:py:class:`GeometryHandler` instance with updated free axes based on the gradient dependencies.

        Notes
        -----
        This method computes the necessary dependencies for gradient computations and creates
        a new:py:class:`GeometryHandler` instance with the relevant free axes.
        """
        return self.__class__(self.coordinate_system,free_axes=self.get_gradient_dependence(axes=axes,basis=basis))

    def get_divergence_descendant(self,
                                  axes: Optional[Union[List[str], str]] = None,
                                  basis: str = 'unit') -> 'GeometryHandler':
        """
        Create a descendant:py:class:`GeometryHandler` instance that includes only the axes necessary for divergence computations.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the divergence computation. If None, defaults to all free axes.
        basis : str, optional
            The basis in which the divergence is computed. This can be one of the following:

            - 'unit' (default): The unit basis.
            - 'contravariant': The contravariant basis.
            - 'covariant': The covariant basis.

        Returns
        -------
        GeometryHandler
            A new:py:class:`GeometryHandler` instance with updated free axes based on the divergence dependencies.

        Notes
        -----
        This method computes the necessary dependencies for divergence computations and creates
        a new:py:class:`GeometryHandler` instance with the relevant free axes.
        """
        return self.__class__(self.coordinate_system,free_axes=self.get_divergence_dependence(axes=axes,basis=basis))

    def get_laplacian_descendant(self,
                                 axes: Optional[Union[List[str], str]] = None) -> 'GeometryHandler':
        """
        Create a descendant:py:class:`GeometryHandler` instance that includes only the axes necessary for Laplacian computations.

        Parameters
        ----------
        axes : Optional[Union[List[str], str]], optional
            The axes to consider for the Laplacian computation. If None, defaults to all free axes.

        Returns
        -------
        GeometryHandler
            A new:py:class:`GeometryHandler` instance with updated free axes based on the Laplacian dependencies.

        Notes
        -----
        This method computes the necessary dependencies for Laplacian computations and creates
        a new:py:class:`GeometryHandler` instance with the relevant free axes.
        """
        return self.__class__(self.coordinate_system,free_axes=self.get_laplacian_dependence(axes=axes))


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
                stringent grid shape of ``(N_1,N_2,...,N_DIM_FREE,)`` in order to compute the necessary derivatives.

        coordinates : np.ndarray
            Array of coordinates with shape ``(..., NDIM)``, where ``NDIM`` is the number of **free** dimensions. If numerical
            derivatives are necessary, then ``(..., NDIM)`` must be a grid with more stringent shape ``(N_1,...,N_NDIM,N_DIM)``,
            where ``N_i`` may be any number of grid points.

            .. note::

                The free dimensions are those which are free **after** the operation, not before it. Thus,
                if a gradient computation breaks symmetry, the coordinates must include the now-free coordinates.

        axes : Union['AxisAlias', List['AxisAlias']]
            The axes along which to compute the gradient. This may be any subset of the free axes.
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
        # VALIDATION: The only difference between this and the coordinate system case is that
        # we need to manage the inputs. This requires filling the coordinates in and then
        # passing through a new (fixed) function.
        #
        # Determine the dependence of this operation. This is necessary to ensure that
        # we grab the correct number of coordinates.
        dependent_axes = self.get_gradient_dependence(axes=axes,basis=basis)
        independent_axes = [ax for ax in self.coordinate_system.AXES if ax not in dependent_axes]

        # MANAGE the number of axes
        if axes is None:
            axes = self.free_axes[:]

        # Constructing the new coordinates.
        # This moves (..., NDIM_SIM) -> (..., NDIM)
        # If we have derivatives unspecified and the field is not callable, then we require a grid.
        if (not callable(field)) and (derivatives is None):
            coordinates = self.coerce_coordinate_grid(coordinates, self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))
        else:
            coordinates = self.coerce_coordinates(coordinates,self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))


        # COERCING the field. If the field is callable, it needs to go to a full coordinate function.
        if isinstance(field, Callable):
            field = self.coerce_function(field,self.coordinate_system,self.free_axes)
        if isinstance(field, np.ndarray):
            field = field.reshape((*coordinates.shape[:-1],))

        # COERCING the derivatives. If the derivatives are provided as ndarray, then they should match the shape of
        # the coordinates up to the final axis and then have axes values. If its a list of functions, we need to coerce
        # each function.
        if derivatives is not None:
            if isinstance(derivatives, list):
                # These are all functions.
                derivatives = [self.coerce_function(derivative,self.coordinate_system,self.free_axes) for derivative in derivatives]
            elif isinstance(derivatives, np.ndarray):
                derivatives = np.reshape(derivatives, (*coordinates.shape[:-1],len(axes)))

        # pass to coordinate system
        return self.coordinate_system.compute_gradient(field,coordinates,axes=axes,derivatives=derivatives,basis=basis,**kwargs)


    def compute_divergence(self,
                           field: Union[np.ndarray, Callable],
                           axes: Union['AxisAlias', List['AxisAlias']],
                           /,
                           coordinates: Optional[np.ndarray] = None,
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
        axes : Union['AxisAlias', List['AxisAlias']]
            The axes along which the field components are defined.
        coordinates : np.ndarray, optional
            The coordinates over which the divergence should be computed. In general, these may be ``(...,NDIM_FREE)`` in shape;
            however, if ``derivatives = None`` and the ``field`` is an ``np.ndarray``, then the coordinates must be a proper
            coordinate grid with generic shape ``(N_1,...,N_NDIM_FREE, NDIM_FREE)`` where each ``N_i`` may be any integer corresponding
            to the number of points along the grid in that dimension.
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
        # VALIDATION: The only difference between this and the coordinate system case is that
        # we need to manage the inputs. This requires filling the coordinates in and then
        # passing through a new (fixed) function.
        #
        # Determine the dependence of this operation. This is necessary to ensure that
        # we grab the correct number of coordinates.
        dependent_axes = self.get_divergence_dependence(axes=axes,basis=basis)
        independent_axes = [ax for ax in self.coordinate_system.AXES if ax not in dependent_axes]

        # MANAGE the number of axes
        if axes is None:
            axes = self.free_axes[:]

        # Constructing the new coordinates.
        # This moves (..., NDIM_SIM) -> (..., NDIM)
        # If we have derivatives unspecified and the field is not callable, then we require a grid.
        if (not callable(field)) and (derivatives is None):
            coordinates = self.coerce_coordinate_grid(coordinates, self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))
        else:
            coordinates = self.coerce_coordinates(coordinates,self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))


        # COERCING the field. If the field is callable, it needs to go to a full coordinate function.
        if isinstance(field, Callable):
            field = self.coerce_function(field,self.coordinate_system,self.free_axes)
        if isinstance(field, np.ndarray):
            field = field.reshape((*coordinates.shape[:-1],len(axes)))

        # COERCING the derivatives. If the derivatives are provided as ndarray, then they should match the shape of
        # the coordinates up to the final axis and then have axes values. If its a list of functions, we need to coerce
        # each function.
        if derivatives is not None:
            if isinstance(derivatives, list):
                # These are all functions.
                derivatives = [self.coerce_function(derivative,self.coordinate_system,self.free_axes) for derivative in derivatives]
            elif isinstance(derivatives, np.ndarray):
                derivatives = np.reshape(derivatives, (*coordinates.shape[:-1],len(axes)))

        # pass to coordinate system
        return self.coordinate_system.compute_divergence(field,coordinates,axes=axes,derivatives=derivatives,basis=basis,**kwargs)


    def compute_laplacian(self,
                          field: Union[np.ndarray, Callable],
                          coordinates: np.ndarray,
                          axes: Union['AxisAlias', List['AxisAlias']],
                          *,
                          basis: str = 'unit',
                          first_derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                          second_derivatives: List[Optional[Union[np.ndarray, Callable]]] = None,
                          **kwargs) -> np.ndarray:
        """
        Compute the Laplacian of a scalar field in the coordinate system.

        Parameters
        ----------
        field : Union[np.ndarray, Callable]
            The scalar field to compute the Laplacian of:
            - A numpy array for numerical input. This must be a ``(...,)`` array matching the shape of
              the `coordinates` argument (up to the final dimension).
            - A callable function returning a numpy array for functional input. This should have the signature
              ``f(x_0, ..., x_n)`` and return an array of the same shape as each of the input coordinates.

        coordinates : np.ndarray
            The array of coordinates over which the Laplacian is computed. These should have a shape of
            ``(..., NDIM_FREE)``, where `NDIM_FREE` is the number of free dimensions.

        axes : Union['AxisAlias', List['AxisAlias']]
            The axes to consider for the Laplacian computation. These must correspond to free axes of the coordinate system.

        basis : {'unit', 'covariant', 'contravariant'}, optional
            The basis in which the Laplacian is computed. Default is 'unit'. Supported options:
            - 'unit': The unit basis.
            - 'covariant': The covariant basis.
            - 'contravariant': The contravariant basis.

        first_derivatives : List[Optional[Union[np.ndarray, Callable]]], optional
            Precomputed first derivatives of the field along each axis. If provided:
            - For `Callable` fields: This should be a list of callables, each computing the first derivative along a specific axis.
            - For `np.ndarray` fields: This should be a list of arrays matching the shape of `coordinates`.
            If None, numerical differentiation will be performed.

        second_derivatives : List[Optional[Union[np.ndarray, Callable]]], optional
            Precomputed second derivatives of the field along each axis. If provided:
            - For `Callable` fields: This should be a list of callables, each computing the second derivative along a specific axis.
            - For `np.ndarray` fields: This should be a list of arrays matching the shape of `coordinates`.
            If None, numerical differentiation will be performed.

        **kwargs
            Additional keyword arguments for numerical differentiation.

        Returns
        -------
        np.ndarray
            The computed Laplacian of the scalar field. The output will have the same shape as the input `coordinates`
            up to the final axis.

        Notes
        -----
        The Laplacian of a scalar field :math:`f` in orthogonal coordinates is given by:

        .. math::

            \nabla^2 f = \sum_k \left( \frac{1}{\lambda_k^2} \frac{\partial^2 f}{\partial x_k^2} + L_k \frac{\partial f}{\partial x_k} \right),

        where :math:`\lambda_k` is the Lame coefficient for the :math:`k`-th axis, and :math:`L_k` is the Laplacian term
        specific to the coordinate geometry.

        If `first_derivatives` or `second_derivatives` are not provided, numerical differentiation will be performed
        on the input field and coordinates.
        """
        # VALIDATION: The only difference between this and the coordinate system case is that
        # we need to manage the inputs. This requires filling the coordinates in and then
        # passing through a new (fixed) function.
        #
        # Determine the dependence of this operation. This is necessary to ensure that
        # we grab the correct number of coordinates.
        dependent_axes = self.get_gradient_dependence(axes=axes,basis=basis)
        independent_axes = [ax for ax in self.coordinate_system.AXES if ax not in dependent_axes]
        original_coordinate_shape = coordinates.shape
        # MANAGE the number of axes
        if axes is None:
            axes = self.free_axes[:]

        # Constructing the new coordinates.
        # This moves (..., NDIM_SIM) -> (..., NDIM)
        # If we have derivatives unspecified and the field is not callable, then we require a grid.
        if (first_derivatives is None) or (second_derivatives is None):
            coordinates = self.coerce_coordinate_grid(coordinates, self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))
        else:
            coordinates = self.coerce_coordinates(coordinates,self.coordinate_system, dependent_axes, list(self.get_fill_values(independent_axes)))

        # COERCING the field. If the field is callable, it needs to go to a full coordinate function.
        if isinstance(field, Callable):
            field = self.coerce_function(field,self.coordinate_system,self.free_axes)
        if isinstance(field, np.ndarray):
            field = field.reshape((*coordinates.shape[:-1],))

        # COERCING the derivatives. If the derivatives are provided as ndarray, then they should match the shape of
        # the coordinates up to the final axis and then have axes values. If its a list of functions, we need to coerce
        # each function.
        if first_derivatives is not None:
            if isinstance(first_derivatives, list):
                # These are all functions.
                first_derivatives = [self.coerce_function(derivative,self.coordinate_system,self.free_axes) for derivative in first_derivatives]
            elif isinstance(first_derivatives, np.ndarray):
                first_derivatives = np.reshape(first_derivatives, (*coordinates.shape[:-1],len(axes)))
        
        # COERCING the 2nd derivatives. If the derivatives are provided as ndarray, then they should match the shape of
        # the coordinates up to the final axis and then have axes values. If its a list of functions, we need to coerce
        # each function.
        if second_derivatives is not None:
            if isinstance(second_derivatives, list):
                # These are all functions.
                second_derivatives = [self.coerce_function(derivative,self.coordinate_system,self.free_axes) for derivative in second_derivatives]
            elif isinstance(second_derivatives, np.ndarray):
                second_derivatives = np.reshape(second_derivatives, (*coordinates.shape[:-1],len(axes)))

        laplacian = self.coordinate_system.compute_laplacian(field,coordinates,axes=axes,basis=basis,first_derivatives=first_derivatives,second_derivatives=second_derivatives,**kwargs)
        return laplacian.reshape(original_coordinate_shape[:-1])

    @property
    def free_axes(self) -> List[str]:
        """
        List of axes that are not affected by symmetry in the geometry handler.

        These axes are considered free for operations and are preserved during
        computations.

        Returns
        -------
        List[str]
            The list of free axes defined in the coordinate system.

        Notes
        -----
        - Free axes are specified during initialization or default to all axes
          in the coordinate system if none are specified.
        """
        return self._free_axes

    @property
    def symmetric_axes(self) -> List[str]:
        """
        List of axes that are affected by symmetry in the geometry handler.

        These axes are considered fixed and will be assigned default fill values
        during computations.

        Returns
        -------
        List[str]
            The list of symmetric (fixed) axes defined in the coordinate system.

        Notes
        -----
        - Symmetric axes are determined as the complement of the free axes.
        """
        return self._sym_axes

    @property
    def NDIM(self) -> int:
        """
        Total number of dimensions in the associated coordinate system.

        Returns
        -------
        int
            The total number of dimensions (`NDIM`) of the coordinate system.
        """
        return self.coordinate_system.NDIM

    @property
    def NDIM_FREE(self) -> int:
        """
        Number of free dimensions in the geometry handler.

        Free dimensions are those not affected by symmetry and are preserved
        during computations.

        Returns
        -------
        int
            The number of free axes.

        Notes
        -----
        - This is equivalent to the length of `free_axes`.
        """
        return len(self._free_axes)

    @property
    def NDIM_SYM(self) -> int:
        """
        Number of symmetric dimensions in the geometry handler.

        Symmetric dimensions are those affected by symmetry and are assigned
        default fill values during computations.

        Returns
        -------
        int
            The number of symmetric (fixed) axes.

        Notes
        -----
        - This is equivalent to the length of `symmetric_axes`.
        """
        return len(self._sym_axes)

    @property
    def symmetry_mask(self) -> np.ndarray[bool]:
        """
        Boolean mask indicating the symmetric axes.

        A cached array where each element corresponds to an axis in the coordinate
        system, and the value is `True` if the axis is symmetric (fixed), `False` otherwise.

        Returns
        -------
        np.ndarray
            A boolean array of shape `(NDIM,)` indicating the symmetric axes.

        Notes
        -----
        - This mask is lazily initialized and cached for future use.
        - The mask aligns with the order of axes in the coordinate system.
        """
        if self._symmetric_mask is None:
            self._symmetric_mask = np.array([ax in self._sym_axes for ax in self.coordinate_system.AXES], dtype=bool)

        return self._symmetric_mask

    @property
    def free_mask(self) -> np.ndarray[bool]:
        """
        Boolean mask indicating the free axes.

        A cached array where each element corresponds to an axis in the coordinate
        system, and the value is `True` if the axis is free, `False` otherwise.

        Returns
        -------
        np.ndarray
            A boolean array of shape `(NDIM,)` indicating the free axes.

        Notes
        -----
        - This mask is lazily initialized and cached for future use.
        - The mask aligns with the order of axes in the coordinate system.
        """
        if self._free_mask is None:
            self._free_mask = np.array([ax not in self._sym_axes for ax in self.coordinate_system.AXES], dtype=bool)

        return self._free_mask