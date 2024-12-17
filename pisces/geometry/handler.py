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
from pisces.utilities.array_utils import CoordinateArray

if TYPE_CHECKING:
    from pisces.geometry._typing import AxisAlias

class GeometryHandler:

    def __init__(self, coordinate_system: CoordinateSystem,
                 /,
                 free_axes: Optional[List[str]] = None,
                 *,
                 fill_values: Optional[float|List[float]] = 0.0):
        # CONFIGURING the core attributes.
        # Set the coordinate system, identify the relevant symmetry axes and then
        # proceed with initialization.
        self.coordinate_system = coordinate_system

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
            A subset of `self._sym_axes` for which to fetch the fill values. If None, fetches all
            fill values for `self._sym_axes`. Default is None.

        Returns
        -------
        np.ndarray
            An array of fill values corresponding to the specified axes.

        Raises
        ------
        ValueError
            If any axis in `fixed_axes` is not in `self._sym_axes`.
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
            The partial coordinates array with shape (..., NDIM_FREE), where NDIM_FREE is the number of free axes.
        coordinate_system : CoordinateSystem
            The coordinate system to use for coercion, defining the full set of axes.
        free_axes : List[str]
            The subset of axes considered free, corresponding to the provided `coordinates`.
        fill_values : float or List[float], optional
            The fill values for the symmetric (non-free) axes. Default is 0.0.

        Returns
        -------
        np.ndarray
            The full coordinates array with shape (..., NDIM), where NDIM is the number of dimensions in the coordinate system.

        Raises
        ------
        ValueError
            If the free axes are not valid, or if the shape of `coordinates` or `fill_values` does not match expectations.
        """
        # Validate the free axes
        if any(fa not in coordinate_system.AXES for fa in free_axes):
            raise ValueError(
                f"Axes in {free_axes} are not defined in the coordinate system (axes={coordinate_system.AXES})")

        # Identify fixed (symmetric) axes
        fixed_axes = [ax for ax in coordinate_system.AXES if ax not in free_axes]
        free_mask = np.array([ax in free_axes for ax in coordinate_system.AXES], dtype=bool)
        fixed_mask = ~free_mask

        # Manage the fill values
        if isinstance(fill_values, float):
            fill_values = [fill_values] * len(fixed_axes)

        if len(fill_values) != len(fixed_axes):
            raise ValueError(f"fill_values must have the same length as the number of fixed axes ({len(fixed_axes)})")

        # Ensure coordinates match the expected shape
        try:
            coordinates = CoordinateArray(coordinates, len(free_axes))
        except Exception as e:
            raise ValueError(f"Could not coerce coordinates to have shape (..., {len(free_axes)}).\n"
                             f"Error: {e}\n"
                             f"This likely means that you are performing an operation which requires additional axes to"
                             f" be specified because of symmetry breaking...\nThe fixed axes are {fixed_axes} and the free "
                             f" axes are {free_axes}.") from e

        # Construct the full coordinates array
        full_coordinates = np.empty((*coordinates.shape[:-1], coordinate_system.NDIM))
        full_coordinates[..., free_mask] = coordinates
        full_coordinates[..., fixed_mask] = np.broadcast_to(fill_values, full_coordinates[..., fixed_mask].shape)

        return full_coordinates

    @classmethod
    def coerce_coordinate_grid(cls,
                               coordinates: np.ndarray,
                               coordinate_system: CoordinateSystem,
                               free_axes: List[str],
                               fill_values: float | List[float] = 0.0) -> np.ndarray:
        # COERCE the original coordinates.
        coordinates = cls.coerce_coordinates(coordinates, coordinate_system, free_axes, fill_values)

        # DETERMINE the correct shape.
        grid_shape = [coordinates.shape[free_axes.index(ax)] if ax in free_axes else 1 for ax in coordinate_system.AXES]

        # FIX the missing axes.
        return coordinates.reshape((*grid_shape, coordinates.shape[-1]))

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
        coordinate_system : CoordinateSystem
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
            The axes to consider for the gradient computation. If None, defaults to `self._free_axes`.
        basis : str, optional
            The basis in which the gradient is computed (not currently used in this method but included for future compatibility).
            Default is 'unit'.

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
        return self.__class__(self.coordinate_system,free_axes=self.get_gradient_dependence(axes=axes,basis=basis))

    def get_divergence_descendant(self,
                                  axes: Optional[Union[List[str], str]] = None,
                                  basis: str = 'unit') -> 'GeometryHandler':
        return self.__class__(self.coordinate_system,free_axes=self.get_divergence_dependence(axes=axes,basis=basis))

    def get_laplacian_descendant(self,
                                 axes: Optional[Union[List[str], str]] = None) -> 'GeometryHandler':
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
    def free_axes(self):
        return self._free_axes

    @property
    def symmetric_axes(self):
        return self._sym_axes

    @property
    def NDIM(self):
        return self.coordinate_system.NDIM

    @property
    def NDIM_FREE(self):
        return len(self._free_axes)

    @property
    def NDIM_SYM(self):
        return len(self._sym_axes)

    @property
    def symmetry_mask(self):
        if self._symmetric_mask is None:
            self._symmetric_mask = np.array([ax in self._sym_axes for ax in self.coordinate_system.AXES], dtype=bool)

        return self._symmetric_mask

    @property
    def free_mask(self):
        if self._free_mask is None:
            self._free_mask = np.array([ax not in self._sym_axes for ax in self.coordinate_system.AXES], dtype=bool)

        return self._free_mask
