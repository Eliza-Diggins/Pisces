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
from typing import Optional, List, Collection, Any, Union, Callable

from pisces.geometry.base import CoordinateSystem
from pisces.geometry.symmetry import Symmetry
from pisces.utilities.array_utils import fill_missing_coord_axes, complete_and_reshape_as_grid, is_grid
from pisces.utilities.general import find_in_subclasses
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray
import h5py


# noinspection PyProtectedMember
@dataclass
class GeometryHandler:
    r"""
    Core class for managing computations in specific coordinate systems with a given
    symmetry.

    Effectively, the :py:class:`GeometryHandler` class combines the :py:class:`~pisces.geometry.symmetry.Symmetry` and
    :py:class:`~pisces.geometry.base.CoordinateSystem` classes into a single self-consistent operation handler. Thus,
    the geometry handler manages the array manipulations and symmetry implications for any operations performed
    behind the scenes in Pisces.

    Attributes
    ----------
    coordinate_system : :py:class:`~pisces.geometry.base.CoordinateSystem`
        The coordinate system associated with the geometry. All array operations performed by the
        :py:class:`GeometryHandler` are handled by this coordinate system.
    symmetry : :py:class:`~pisces.geometry.symmetry.Symmetry`, optional
        Symmetry associated with the coordinate system, which determines invariance across specific axes.
        If None, an empty symmetry is assigned by default.
    fill_values : NDArray[float], default=0.0
        Default values for the symmetric axes of the coordinate system. If provided, these should be an array
        with length equal to ``symmetry``'s :py:attr:`~pisces.geometry.symmetry.Symmetry.NDIM_SIM` attribute. When
        a computation is performed, the symmetric axes (which are not provided in the coordinates) are generated
        using these fill values.

        .. note::

            These should be provided in the order that the axes appear in ``coordinate_system``.

        By default, ``fill_values`` is ``0.0``, which will cause every missing axis to be filled by ``0``.

    Examples
    --------

    For this example, we will utilize a set of oblate homoeoidal coordinates with radial symmetry to perform a set
    of basic vector calculations. The first things to do is initialize the symmetry and the coordinate system:

    >>> from pisces.geometry import Symmetry, OblateHomoeoidalCoordinateSystem
    >>> cs = OblateHomoeoidalCoordinateSystem(ecc = 0.7)
    >>> sym = Symmetry(['phi','theta'],coordinate_system=cs)

    With the symmetry prepared, we're ready to create the :py:class:`GeometryHandler` object:

    >>> handler = GeometryHandler(cs,sym)
    >>> handler
    <GeometryHandler(axes={1, 2},cs=<OblateHomoeoidalCoordinateSystem(ecc=0.7)>)>

    Let's put this to the test! Let's assume that we have a function :math:`f(r) = r^2`. If we attempt to compute the
    gradient of this function using the :py:class:`~pisces.geometry.base.CoordinateSystem`, we get an error:

    >>> r = np.linspace(0,1,100)
    >>> func = lambda x: x**2
    >>> y = handler.coordinate_system.compute_function_gradient_term(r,func,axis=0) # doctest: +SKIP

    .. code-block:: shell

        File "pisces/geometry/handlers.py", line 79, in GeometryHandler
        Failed example:
            y = handler.coordinate_system.compute_function_gradient_term(r,func,axis=0)
        Exception raised:
            Traceback (most recent call last):
              File "<doctest GeometryHandler[7]>", line 1, in <module>
                y = handler.coordinate_system.compute_function_gradient_term(r,func,axis=0)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              File "/pisces/geometry/base.py", line 826, in compute_function_gradient_term
                raise ValueError(f"coordinates have shape `{coordinates.shape}`"
            ValueError: coordinates have shape `(100,)` [ndim=1, end_size=100], but expected shape like `(...,3)` [ndim=4, end_size=3].

    What this is telling us is that our coordinates (which were just an array of 100 radii) aren't a valid grid
    shape for the coordinate system because the coordinate system expects a grid with a shape like ``(N_r,N_theta,N_phi,3)``
    instead. Obviously, it's a painful endeavour to fill in those arrays every time we want to do something. As such, the handler
    does it for us! Let's try it with the **handler** instead of the **coordinate system**:

    >>> r = np.linspace(0,1,100)
    >>> func = lambda x: x**2
    >>> y = handler.compute_function_gradient_term(r,func,axis=0) # doctest: +SKIP

    .. code-block:: shell

        Error
        **********************************************************************
        File "/pisces/geometry/handlers.py", line 102, in GeometryHandler
        Failed example:
            y = handler.compute_function_gradient_term(r,func,axis=0)
        Exception raised:
              File "<doctest GeometryHandler[10]>", line 1, in <module>
                y = handler.compute_function_gradient_term(r,func,axis=0)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              File "/pisces/geometry/handlers.py", line 685, in compute_function_gradient_term
                raise ValueError(f"Validation Failed: {e}") from e
            ValueError: Validation Failed: Coordinate shape was neither the full coordinate dimension (3) or the reduced asymmetric dimension (2), thus the coordinate values could not be filled.

    **It's still telling us we have the wrong number of dimensions!**

    .. hint::

        This is happening because the coordinate system breaks the symmetry when operating with the gradient! Let's see:

        >>> handler.symmetry
        <Symmetry: axes=[1, 2], cs=<OblateHomoeoidalCoordinateSystem(ecc=0.7)>>
        >>> handler.symmetry.gradient(active_axes=[0])
        <Symmetry: axes=[2], cs=<OblateHomoeoidalCoordinateSystem(ecc=0.7)>>

        So we've lost our :math:`\theta` invariance. We'll need to provide :math:`\theta` coordinates as well!

    **Let's give it a final go!**
    This time, we're going to provide a grid of :math:`\theta` values on which to compute the gradient.

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry import Symmetry,GeometryHandler, OblateHomoeoidalCoordinateSystem
        >>> cs = OblateHomoeoidalCoordinateSystem(ecc = 0.7)
        >>> sym = Symmetry(['phi','theta'],coordinate_system=cs)
        >>> handler = GeometryHandler(cs,sym)
        >>> func = lambda x: x**2
        >>> coordinates = np.mgrid[0:1:100j,0:np.pi:100j]
        >>> coordinates = np.moveaxis(coordinates, 0, -1)
        >>> grad_r_field = handler.compute_function_gradient_term(coordinates,func,axis=0)
        >>> im = plt.imshow(grad_r_field[0,...].T,extent=(0,np.pi,0,1),cmap='inferno')
        >>> _ = plt.xlabel(r"$r$")
        >>> _ = plt.ylabel(r"$\theta$")
        >>> _ = plt.colorbar(im,orientation='horizontal', label=r"$\nabla(r^2)$")
        >>> plt.show()

    """
    coordinate_system: 'CoordinateSystem'
    symmetry: Optional['Symmetry'] = None
    fill_values: NDArray[np.floating] = 0.0

    def __post_init__(self):
        """
        Post-initialization method for GeometryHandler.

        Sets up symmetry if not provided, coerces fill values to the appropriate shape, and prepares for
        symmetry operations. Default fill values are reshaped as needed to match the dimensionality.
        """
        # Set symmetry to empty if not provided
        if self.symmetry is None:
            self.symmetry = Symmetry.empty_symmetry(self.coordinate_system)

        # Ensure fill_values is correctly shaped
        if not isinstance(self.fill_values, Collection):
            # noinspection PyTypeChecker
            self.fill_values = [self.fill_values] * self.symmetry.NDIM_SIM
        self.fill_values = np.array(self.fill_values).reshape((self.symmetry.NDIM_SIM,))

    def __str__(self):
        return f"<GeometryHandler(axes={self.symmetry.symmetry_axes},cs={self.coordinate_system})>"

    def __repr__(self):
        return self.__str__()

    def get_fill_values(self, symmetry: 'Symmetry') -> NDArray[np.floating]:
        """
        Retrieve appropriate fill values based on the provided symmetry.

        Parameters
        ----------
        symmetry : Symmetry
            Symmetry object specifying which axes require fill values.

        Returns
        -------
        NDArray[np.floating]
            Fill values corresponding to symmetric axes in the provided symmetry.
        """
        mask = np.array(symmetry._invariance_array[self.symmetry._invariance_array], dtype=bool)
        return self.fill_values[mask]

    def fill_missing_coordinates(self, coordinates: NDArray[np.floating], symmetry: Optional['Symmetry'] = None) -> \
    NDArray[np.floating]:
        """
        Fill missing coordinate values based on symmetry and axis mask.

        Parameters
        ----------
        coordinates : NDArray[np.floating]
            The array of known coordinates, shaped `(..., M)` where `M` matches present axes based on symmetry.
        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to `self.symmetry` if not provided.

        Returns
        -------
        NDArray[np.floating]
            A complete coordinate array for all axes, with filled values for missing axes.

        Raises
        ------
        ValueError
            If input coordinate dimensionality does not match expected `symmetry` configuration.

        Notes
        -----
        The missing axes are filled with :py:attr:`GeometryHandler.fill_values`. If the default symmetry is used,
        then these are used as they are. Otherwise, the fill values for each of the symmetric axes in the default
        symmetry which are also symmetric in the new symmetry are pulled and used.
        """
        symmetry = symmetry or self.symmetry
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 1)

        if coordinates.shape[-1] == symmetry.NDIM_NSIM:
            return fill_missing_coord_axes(coordinates, ~symmetry._invariance_array, self.get_fill_values(symmetry))
        elif coordinates.shape[-1] == self.coordinate_system.NDIM:
            return coordinates
        else:
            raise ValueError(f"Coordinate shape was neither the full coordinate dimension ({self.coordinate_system.NDIM}) or "
                             f"the reduced asymmetric dimension ({symmetry.NDIM_NSIM}), thus the coordinate values could not be "
                             f"filled.")

    def coerce_to_grid(self, coordinates: NDArray[np.floating], grid_axis_mask: Optional[NDArray[np.bool_]] = None,
                       symmetry: Optional['Symmetry'] = None) -> NDArray[np.floating]:
        """
        Ensures `coordinates` are arranged as a grid, filling in missing axes and restructuring to match the expected grid format.

        This function takes an input array of coordinates and restructures it into a grid format, adding singleton dimensions
        for missing axes based on the provided `grid_axis_mask`. If the `grid_axis_mask` is not provided, a default will be
        assumed based on the shape of `coordinates` and symmetry.

        Parameters
        ----------
        coordinates : NDArray[np.floating]
            Input array of coordinates, with shape `(N, ndim_present)`, where `ndim_present` is the number of dimensions currently
            represented in the array. This may be a reduced set of axes, depending on symmetry.
        grid_axis_mask : Optional[NDArray[np.bool_]], optional
            A boolean mask specifying which axes in the grid structure should correspond to actual coordinate values in
            `coordinates`. Each `True` value in `grid_axis_mask` indicates that the corresponding axis in `coordinates`
            should be represented as a full dimension in the grid; `False` values will result in singleton dimensions
            for missing axes. If `grid_axis_mask` is `None`, it will be set automatically:

            - If `coordinates` shape matches the number of non-symmetric axes (`symmetry.NDIM_NSIM`), `grid_axis_mask` defaults
              to the inverted symmetry array, marking each non-symmetric axis as `True`.
            - If `coordinates` shape matches the full dimensionality (`self.coordinate_system.NDIM`), all entries in
              `grid_axis_mask` will be set to `True`, and the array is returned unchanged.

            **Example**:

            For a 3D coordinate system where only `x` and `z` coordinates are present (`symmetry` is invariant along `y`):

            - `grid_axis_mask = [True, False, True]` results in a grid with shape `(N_x, 1, N_z)`, where `y` is a singleton dimension.
            - If `grid_axis_mask` is omitted, it will default to `[True, False, True]` based on symmetry, resulting in the same grid.

            This parameter allows flexibility in representing certain axes with singleton dimensions, useful in higher-dimensional
            _grids where some axes are invariant.

        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to `self.symmetry`. This parameter determines the
            underlying structure of the `grid_axis_mask` if it is not provided, as well as how missing coordinates in
            `coordinates` should be filled.

        Returns
        -------
        NDArray[np.floating]
            Reshaped coordinate array with a full grid structure, where missing axes are represented by singleton dimensions
            according to `grid_axis_mask`.

        Raises
        ------
        ValueError
            If the dimensions of `coordinates` do not align with expected symmetry and grid mask.

        """
        symmetry = symmetry or self.symmetry
        coordinates = np.array(coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(-1, 1)

        if coordinates.shape[-1] == symmetry.NDIM_NSIM:
            axes_mask = ~symmetry._invariance_array
        elif coordinates.shape[-1] == self.coordinate_system.NDIM:
            axes_mask = np.ones(self.coordinate_system.NDIM, dtype=bool)
        else:
            raise ValueError("Coordinate shape does not match expected symmetry configuration.")

        if grid_axis_mask is None:
            if (coordinates.ndim - 1) == symmetry.NDIM_NSIM:
                grid_axis_mask = ~symmetry._invariance_array

        return complete_and_reshape_as_grid(coordinates, axis_mask=axes_mask, grid_axis_mask=grid_axis_mask)

    def compute_gradient_term(self,
                              coordinates: NDArray,
                              values: NDArray,
                              axis: int,
                              derivative: Optional[NDArray] = None,
                              basis: str = 'unit',
                              __validate__: bool = True,
                              **kwargs) -> NDArray:
        r"""
        Compute a term of the gradient for a scalar field along a specified axis, adjusted by basis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shaped `(N, NDIM)`.
        values : NDArray
            Scalar field values at specified coordinates.
        axis : int
            Index of the axis for gradient computation.
        derivative : Optional[NDArray], optional
            Precomputed derivative along `axis`. If None, it will be calculated.
        basis : {'unit', 'covariant', 'contravariant'}, default 'unit'
            Basis to apply to the gradient term:
            - 'unit': Scales with the Lame coefficient.
            - 'covariant': Scales with Lame coefficient squared.
            - 'contravariant': Returns direct partial derivative.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional parameters for the derivative computation.

        Returns
        -------
        NDArray
            Gradient term along `axis`, adjusted based on the selected basis.

        Raises
        ------
        ValueError
            If coordinates cannot be coerced into a grid or if validation fails.

        Examples
        --------
        In a spherical coordinate system, compute the radial gradient term:

        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.handlers import GeometryHandler
        >>> cs = SphericalCoordinateSystem()
        >>> handler = GeometryHandler(cs, Symmetry(['phi', 'theta'], cs))
        >>> coords = np.linspace(0, 10, 100).reshape(-1, 1)
        >>> values = coords ** 2
        >>> grad = handler.compute_gradient_term(coords, values, axis=0)

        See Also
        --------
        partial_derivative : Computes axis-wise partial derivatives.
        gradient : Computes the entire gradient for a scalar field.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.gradient_component(axis, basis)
                coordinates = self.coerce_to_grid(coordinates, symmetry=_op_symmetry)

                if not is_grid(coordinates):
                    raise ValueError("Coordinates are not structured as a grid.")
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Delegate gradient computation to the coordinate system with correct basis adjustments
        return self.coordinate_system.compute_gradient_term(coordinates, values, axis, derivative=derivative,
                                                            basis=basis, **kwargs)

    def gradient(self,
                 coordinates: NDArray,
                 values: NDArray,
                 derivatives: NDArray = None,
                 /,
                 basis='unit',
                 active_axes: List[int] | None = None,
                 __validate__: bool = True,
                 **kwargs):
        """
        Compute the gradient of a scalar field with respect to coordinates in a specified basis.

        This method calculates the gradient of ``values`` (a scalar field) with respect to ``coordinates``
        in a given coordinate system, accounting for different bases: unit, covariant, or contravariant.
        The gradient is scaled by the appropriate Lame coefficients if required by the basis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points, shape ``(N,NDIM)``, where ```N``` is the number of points, and ``NDIM``
            is the number of dimensions in the coordinate system.
        values : NDArray
            Array of scalar field values at the specified coordinates, with shape ``(N,)`` or ``(N, 1)``.
        derivatives : Optional[NDArray], optional
            Array of precomputed partial derivatives along all axes, with shape ``(N, NDIM)``. If ``None``,
            the partial derivatives are calculated within the function.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which to compute the gradient. Default is ``'unit'``:
            - ``'unit'``: Returns derivatives scaled by the Lame coefficients.
            - ``'covariant'``: Returns derivatives scaled by the square of the Lame coefficients.
            - ``'contravariant'``: Returns unscaled derivatives, ignoring Lame coefficients.
        active_axes : Optional[List[int]], optional
            List of active axis indices for which to compute the gradient. If ``None``, all axes are active.
            This parameter allows for selective gradient computation, useful when certain axes can be
            ignored due to symmetry, improving efficiency.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional keyword arguments passed to ``partial_derivatives_all_axes``, such as ``edge_order``.

        Returns
        -------
        NDArray
            Array representing the gradient of ``values`` with respect to ``coordinates``, shaped ``(N, NDIM)``.
            Each row represents the partial derivative along one axis, scaled according to the specified basis.
            If ``active_axes`` is specified, then the return shape will be ``(N,len(active_axes))``.

        Notes
        -----
        The gradient in a non-Cartesian coordinate system depends on the Lame coefficients :math:`h_i`
        for each coordinate axis. The different bases adjust the gradient as follows:

        - **Contravariant basis**: Direct partial derivatives, ignoring Lame coefficients.
        - **Unit basis**: Partial derivatives scaled by the Lame coefficients :math:`h_i`.
        - **Covariant basis**: Partial derivatives scaled by :math:`h_i^2`.

        This flexibility allows for efficient and accurate gradient calculations, accounting for the
        specific requirements of the coordinate system.

        Raises
        ------
        ValueError
            If ``coordinates`` and ``values`` have mismatched shapes, or if ``coordinates`` and ``derivatives``
            shapes mismatch, or if an unrecognized ``basis`` value is provided.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.gradient(basis=basis, active_axes=active_axes)
                coordinates = self.coerce_to_grid(coordinates, symmetry=_op_symmetry)

                if not is_grid(coordinates):
                    raise ValueError("Coordinates are not structured as a grid.")
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Computing
        return self.coordinate_system.gradient(coordinates,
                                               values,
                                               derivatives=derivatives,
                                               basis=basis,
                                               **kwargs)

    def compute_divergence_term(self, coordinates: NDArray,
                                vector_field: NDArray,
                                axis: int,
                                /,
                                basis='unit',
                                __validate__: bool = True,
                                **kwargs):
        r"""
        Compute the contribution to the divergence from a single axis in a specified basis.

        Given that the divergence takes the form

        .. math::

            \nabla \cdot {\bf F} = \frac{1}{J}\partial_i\left(\frac{J}{h_i}\hat{F}_i\right),

        this method computes a single term of that sum corresponding to ``axis``.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``, where ``N`` is the number of points
            and ``NDIM`` is the number of dimensions.
        vector_field : NDArray
            Array of vector field components at each coordinate point, with shape ``(N, NDIM)``.
        axis : int
            The index of the axis along which the divergence term is computed.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which *the vector field is provided*. Defaults to ``'unit'``. This will change
            the equation by a scaling factor (the Lame coefficient).

            .. hint::

                In most cases, we express vector fields in the ``"unit"`` basis, that way the vectors don't
                implicitly scale at different points in space (as they would if covariant or contravariant); however,
                there may be some instances in which this kwarg must be set.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional arguments to be passed to ``partial_derivative``, such as ``edge_order``.

        Returns
        -------
        NDArray
            The computed divergence term along the specified axis, with shape ``(P,)``.

        Notes
        -----
        In many cases, the Lame Coefficients may not depend on the axis specified and therefore, the
        equation simplifies somewhat. To handle this, the :py:attr:`CoordinateSystem.lame_dependence_matrix` is used
        to determine what lame coefficients actually matter for the computation and only those are used.

        Additionally, this method is left public to ensure that users who might need to utilize a specific symmetry,
        which knocks out specific terms in the sum are able to do so efficiently.

        See Also
        --------
        divergence
        """

        if __validate__:
            try:
                _op_symmetry = self.symmetry.divergence_component(axis, basis)
                coordinates = self.coerce_to_grid(coordinates, symmetry=_op_symmetry)

                if not is_grid(coordinates):
                    raise ValueError("Coordinates are not structured as a grid.")
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e


        # Computing
        return self.coordinate_system.compute_gradient_term(coordinates,
                                                            vector_field,
                                                            axis,
                                                            basis=basis,
                                                            **kwargs)

    def divergence(
            self,
            coordinates: NDArray,
            vector_field: NDArray,
            /,
            basis: str = 'unit',
            active_axes: Optional[List[int]] = None,
            __validate__: bool = True,
            **kwargs
    ) -> NDArray:
        r"""
        Compute the divergence of a vector field in this coordinate system.

        The divergence of a vector field :math:`\mathbf{F}` in a coordinate system with
        Lame coefficients is given by:

        .. math::

            \nabla \cdot \mathbf{F} = \sum_i \frac{1}{J} \partial_i \left(\frac{J}{h_i} \hat{F}_i \right),

        where :math:`J` is the Jacobian and :math:`h_i` are the Lame coefficients corresponding
        to each coordinate direction. This method computes the divergence by summing the
        contribution from each axis, and allows basis adjustment to ensure accurate representation.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``, where ``N`` is the number of points
            and ``NDIM`` is the number of dimensions.
        vector_field : NDArray
            Array of vector field components along each axis at each point, with shape ``(N, NDIM)``.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which *the vector field is provided*. Defaults to ``'unit'``:
            - ``'unit'``: Scales the vector field to avoid implicit scaling due to space curvature.
            - ``'covariant'``: Scales by :math:`h_i^2` for each component.
            - ``'contravariant'``: No scaling, assumes direct contravariant basis.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.

        Returns
        -------
        NDArray
            The divergence of the vector field at each coordinate point, with shape ``(N,)``.

        Notes
        -----

        The calculation utilizes the **Lame dependence matrix** to minimize computational load
        by identifying which Lame coefficients depend on the differentiation axis. Only necessary
        terms are computed to optimize performance.

        This method sums contributions from each axis, using ``compute_divergence_term`` to compute
        the divergence term for each specified axis.

        See Also
        --------
        compute_divergence_term : Computes the individual divergence term for a single axis.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.divergence(basis=basis, active_axes=active_axes)
                coordinates = self.coerce_to_grid(coordinates, symmetry=_op_symmetry)

                if not is_grid(coordinates):
                    raise ValueError("Coordinates are not structured as a grid.")
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Computing
        return self.coordinate_system.divergence(coordinates,
                                                 vector_field,
                                                 basis=basis,
                                                 active_axes=active_axes,
                                                 **kwargs)

    def laplacian(
            self,
            coordinates: NDArray,
            scalar_field: NDArray,
            /,
            active_axes: Optional[List[int]] = None,
            __validate__: bool = True,
            **kwargs
    ) -> NDArray:
        r"""
        Compute the Laplacian of a scalar field in this coordinate system.

        The Laplacian :math:`\nabla^2 \phi` of a scalar field :math:`\phi` in an orthogonal
        coordinate system with Lame coefficients :math:`h_i` is given by the divergence of the
        gradient:

        .. math::

            \nabla^2 \phi = \sum_i \frac{1}{J} \partial_i \left( \frac{J}{h_i^2} \frac{\partial \phi}{\partial q^i} \right),

        where :math:`J` is the Jacobian of the transformation. This method computes the Laplacian
        by first calculating the gradient in the specified basis, then applying the divergence to
        this gradient.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``, where ``N`` is the number of points
            and ``NDIM`` is the number of dimensions.
        scalar_field : NDArray
            Array of scalar field values at each coordinate point, with shape ``(N,)``.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the Laplacian calculation. By default, all axes are included.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        kwargs :
            Additional kwargs to pass to :py:meth:`CoordinateSystem.gradient` and to :py:meth:`CoordinateSystem.divergence`.

        Returns
        -------
        NDArray
            The Laplacian of the scalar field at each coordinate point, with shape ``(N,)``.

        Notes
        -----

        **Efficiency and Dependence Matrix**

        This method leverages the Lame dependence matrix for efficient computation by excluding
        unnecessary terms, which further optimizes performance, especially for high-dimensional
        systems.

        See Also
        --------
        divergence : Computes the divergence, used here on the gradient of the scalar field.
        gradient : Computes the gradient of the scalar field prior to divergence.
        """
        # Validate inputs to ensure coordinates and vector field have compatible shapes
        # Determine the operator symmetry

        if __validate__:
            try:
                _op_symmetry = self.symmetry.laplacian(active_axes=active_axes)
                coordinates = self.coerce_to_grid(coordinates, symmetry=_op_symmetry)

                if not is_grid(coordinates):
                    raise ValueError("Coordinates are not structured as a grid.")
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e
        # Computing
        return self.coordinate_system.laplacian(coordinates,
                                                scalar_field,
                                                active_axes=active_axes,
                                                **kwargs)

    def compute_function_gradient_term(self,
                                       coordinates: NDArray,
                                       function: Callable[[NDArray], NDArray],
                                       axis: int,
                                       derivative: Optional[Callable[[NDArray], NDArray]] = None,
                                       basis: str = 'unit',
                                       __validate__: bool = True,
                                       **kwargs) -> NDArray:
        r"""
        Compute a term of the gradient for a scalar function along a specified axis, adjusted by basis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shaped ``(*, NDIM)``. The first ``N-1`` dimensions of the array
            may be any specific shape; however, the final axis must have ``NDIM`` indices corresponding
            to the ``NDIM`` coordinate variables.

            .. hint::

                This allows for both grid shapes and unstructured coordinates to be used.

        function : Callable[[NDArray], NDArray]
            Scalar function to evaluate at specified coordinates.
        axis : int
            Index of the axis for gradient computation.
        derivative : Optional[Callable[[NDArray], NDArray]], optional
            Precomputed derivative along `axis`. If None, it will be calculated.
        basis : {'unit', 'covariant', 'contravariant'}, default 'unit'
            Basis to apply to the gradient term.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional parameters for derivative computation.

        Returns
        -------
        NDArray
            Gradient term along `axis`, adjusted based on the selected basis.

        Examples
        --------

        In this example, we'll compute the gradient of the function :math:`f(r,\theta) = \sin(\theta) / r` in
        polar coordinates. First off, let's take a look at this function on a plot.

        .. plot::
            :include-source:

            >>> from pisces.geometry import PolarCoordinateSystem, GeometryHandler
            >>> import numpy as np
            >>> coordinate_system = PolarCoordinateSystem()
            >>> handler = GeometryHandler(coordinate_system=coordinate_system)
            >>> handler
            <GeometryHandler(axes=set(),cs=PolarCoordinateSystem())>

            In this case, there is no symmetry in the handler. Let's now create the function:

            >>> funct = lambda _r,_theta: np.sin(_theta)**2/_r

            And now, we'll take a cartesian grid, convert it and evaluate the function:

            >>> grid = np.mgrid[-1:1:100j,-1:1:100j]
            >>> grid = np.moveaxis(grid, 0,-1)
            >>> polar_grid = handler.from_cartesian(grid)
            >>> Z = funct(*np.moveaxis(polar_grid,-1,0))

            If we now look at the plot, we'll get an image of the function:

            >>> import matplotlib.pyplot as plt
            >>> im = plt.imshow(Z.T,vmax=5,vmin=0,extent=(-1,1,-1,1),cmap='inferno',origin='lower')
            >>> _ = plt.colorbar(im, label=r"$\sin^2 \theta / r$")
            >>> plt.show()

        Now, in polar coordinates, the Lame coefficients are 1 and :math:`r`, so the gradient is

        .. math::

            \nabla \left(\frac{\sin^2 \theta}{r}\right) = \frac{2\sin \theta \cos \theta}{r^2} \hat{\theta} - \frac{\sin^2 \theta}{r^2} \hat{r}.

        Let's see how we can do this numerically:

        .. plot::
            :include-source:

            >>> from pisces.geometry import PolarCoordinateSystem, GeometryHandler
            >>> import numpy as np
            >>> coordinate_system = PolarCoordinateSystem()
            >>> handler = GeometryHandler(coordinate_system=coordinate_system)
            >>> handler
            <GeometryHandler(axes=set(),cs=PolarCoordinateSystem())>

            In this case, there is no symmetry in the handler. Let's now create the function:

            >>> funct = lambda _r,_theta: np.sin(_theta)**2/_r

            And now, we'll take a cartesian grid, convert it and evaluate the function:

            >>> grid = np.mgrid[-1:1:100j,-1:1:100j]
            >>> grid = np.moveaxis(grid, 0,-1)
            >>> polar_grid = handler.from_cartesian(grid)
            >>> grad_r = handler.compute_function_gradient_term(polar_grid,funct,0)
            >>> grad_theta = handler.compute_function_gradient_term(polar_grid,funct,1)

            If we now look at the plot, we'll get an image of the function:

            >>> import matplotlib.pyplot as plt
            >>> from matplotlib.colors import Normalize
            >>> fig,axes = plt.subplots(2,2,sharex=True,sharey=True)
            >>> _ = axes[0,0].imshow(grad_r[0,...].T,vmax=5,vmin=-5,extent=(-1,1,-1,1),cmap='seismic',origin='lower')
            >>> _ = axes[0,0].text(-.9,0.8,r'$\nabla_r$ computed')
            >>> _ = axes[0,1].imshow(grad_theta[0,...].T,vmax=5,vmin=-5, extent=(-1,1,-1,1),cmap='seismic',origin='lower')
            >>> r,theta = polar_grid[...,0],polar_grid[...,1]
            >>> _ = axes[0,1].text(-.9,0.8,r'$\nabla_\theta$ computed')
            >>> grad_r_exp = -np.sin(theta)**2/r**2
            >>> grad_theta_exp = (2*np.sin(theta)*np.cos(theta))/r**2
            >>> _ = axes[1,0].imshow(grad_r_exp.T,vmax=5,vmin=-5,extent=(-1,1,-1,1),cmap='seismic',origin='lower')
            >>> _ = axes[1,0].text(-.9,0.8,r'$\nabla_r$ expected')
            >>> _ = axes[1,1].imshow(grad_theta_exp.T,vmax=5,vmin=-5, extent=(-1,1,-1,1),cmap='seismic',origin='lower')
            >>> _ = axes[1,1].text(-.9,0.8,r'$\nabla_\theta$ expected')
            >>> _ = plt.colorbar(plt.cm.ScalarMappable(Normalize(-5,5),cmap='seismic'),ax=axes)
            >>> plt.show()


        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.gradient_component(axis, basis)
                coordinates = self.fill_missing_coordinates(coordinates, symmetry=_op_symmetry)

            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Constructing the modified function to ensure that the function signature can take the correct
        # number of dimensions. We assume that function matches the handler in its signature.
        def wrapper_function(*args):
            # Construct the modified coordinates. These are all of the coordinates in the system.
            mod_coords = np.array(args)
            nec_coords = mod_coords[~self.symmetry._invariance_array]

            return function(*nec_coords)

        return self.coordinate_system.compute_function_gradient_term(
            coordinates, wrapper_function, axis, derivative=derivative, basis=basis, **kwargs
        )

    def function_gradient(self,
                          coordinates: NDArray,
                          function: Callable[[NDArray], NDArray],
                          derivatives: Optional[List[Callable[[NDArray], NDArray]]] = None,
                          basis: str = 'unit',
                          active_axes: List[int] | None = None,
                          __validate__: bool = True,
                          **kwargs) -> NDArray:
        """
        Compute the gradient of a scalar function with respect to coordinates in a specified basis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points, shape ``(N,NDIM)``.
        function : Callable[[NDArray], NDArray]
            Scalar function to evaluate at specified coordinates.
        derivatives : Optional[List[Callable[[NDArray], NDArray]]], optional
            Array of precomputed partial derivatives along all axes.
        basis : {'unit', 'covariant', 'contravariant'}, optional
        active_axes : Optional[List[int]], optional
            List of active axis indices for which to compute the gradient.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional keyword arguments passed to ``partial_derivatives_all_axes``.

        Returns
        -------
        NDArray
            Array representing the gradient of the function with respect to coordinates, shaped ``(N, NDIM)``.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.gradient(basis=basis, active_axes=active_axes)
                coordinates = self.fill_missing_coordinates(coordinates, symmetry=_op_symmetry)
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Constructing the modified function to ensure that the function signature can take the correct
        # number of dimensions. We assume that function matches the handler in its signature.
        def wrapper_function(*args):
            # Construct the modified coordinates. These are all of the coordinates in the system.
            mod_coords = np.array(args)
            nec_coords = mod_coords[~self.symmetry._invariance_array]

            return function(*nec_coords)

        return self.coordinate_system.function_gradient(
            coordinates, wrapper_function, derivatives=derivatives, basis=basis, **kwargs
        )

    def compute_function_divergence_term(self,
                                         coordinates: NDArray,
                                         vector_function: Callable[[NDArray], NDArray],
                                         axis: int,
                                         basis: str = 'unit',
                                         __validate__: bool = True,
                                         **kwargs) -> NDArray:
        """
        Compute the contribution to the divergence from a single axis for a vector function.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``.
        vector_function : Callable[[NDArray], NDArray]
            Vector function to evaluate at specified coordinates.
        axis : int
            The index of the axis for divergence computation.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis for the vector function. Defaults to ``'unit'``.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.
        **kwargs
            Additional arguments for derivative computation.

        Returns
        -------
        NDArray
            Computed divergence term for the vector function along the specified axis.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.divergence_component(axis, basis)
                coordinates = self.fill_missing_coordinates(coordinates, symmetry=_op_symmetry)
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Constructing the modified function to ensure that the function signature can take the correct
        # number of dimensions. We assume that function matches the handler in its signature.
        def wrapper_function(*args):
            # Construct the modified coordinates. These are all of the coordinates in the system.
            mod_coords = np.array(args)
            nec_coords = mod_coords[~self.symmetry._invariance_array]

            return vector_function(*nec_coords)

        return self.coordinate_system.compute_function_divergence_term(
            coordinates, wrapper_function, axis, basis=basis, **kwargs
        )

    def function_divergence(self,
                            coordinates: NDArray,
                            vector_function: Callable[[NDArray], NDArray],
                            basis: str = 'unit',
                            active_axes: Optional[List[int]] = None,
                            __validate__: bool = True,
                            **kwargs) -> NDArray:
        """
        Compute the divergence of a vector function in this coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``.
        vector_function : Callable[[NDArray], NDArray]
            Vector function to evaluate at specified coordinates.
        basis : {'unit', 'covariant', 'contravariant'}, optional
        active_axes : Optional[List[int]], optional
            List of axis indices for divergence calculation.
        __validate__ : bool, default True
            If True, validates and fills coordinates as needed.

        Returns
        -------
        NDArray
            Divergence of the vector function at each coordinate point.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.divergence(basis=basis, active_axes=active_axes)
                coordinates = self.fill_missing_coordinates(coordinates, symmetry=_op_symmetry)
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Constructing the modified function to ensure that the function signature can take the correct
        # number of dimensions. We assume that function matches the handler in its signature.
        def wrapper_function(*args):
            # Construct the modified coordinates. These are all of the coordinates in the system.
            mod_coords = np.array(args)
            nec_coords = mod_coords[~self.symmetry._invariance_array]

            return vector_function(*nec_coords)

        return self.coordinate_system.function_divergence(
            coordinates, wrapper_function, basis=basis, active_axes=active_axes, **kwargs
        )

    def function_laplacian(self,
                           coordinates: NDArray,
                           scalar_function: Callable[[NDArray], NDArray],
                           active_axes: Optional[List[int]] = None,
                           __validate__: bool = True,
                           **kwargs) -> NDArray:
        """
        Compute the Laplacian of a scalar function in this coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(N, NDIM)``.
        scalar_function : Callable[[NDArray], NDArray]
            Scalar function to evaluate at specified coordinates.
        active_axes : Optional[List[int]], optional
        __validate__ : bool, default True
        kwargs :
            Additional arguments passed to :py:meth:`CoordinateSystem.function_gradient` and :py:meth:`CoordinateSystem.function_divergence`.

        Returns
        -------
        NDArray
            The Laplacian of the scalar function at each coordinate point.
        """
        if __validate__:
            try:
                _op_symmetry = self.symmetry.laplacian(active_axes=active_axes)
                coordinates = self.fill_missing_coordinates(coordinates, symmetry=_op_symmetry)
            except Exception as e:
                raise ValueError(f"Validation Failed: {e}") from e

        # Constructing the modified function to ensure that the function signature can take the correct
        # number of dimensions. We assume that function matches the handler in its signature.
        def wrapper_function(*args):
            # Construct the modified coordinates. These are all of the coordinates in the system.
            mod_coords = np.array(args)
            nec_coords = mod_coords[~self.symmetry._invariance_array]

            return scalar_function(*nec_coords)

        return self.coordinate_system.function_laplacian(
            coordinates, wrapper_function, active_axes=active_axes, **kwargs
        )

    def from_cartesian(self, coordinates: NDArray, symmetry: Symmetry=None) -> NDArray:
        """
        Convert Cartesian coordinates to the native coordinate system, showing only non-symmetric coordinates.

        Parameters
        ----------
        coordinates : NDArray
            Cartesian coordinates, shaped ``(..., 3)``.
        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to the symmetry of this handler.

        Returns
        -------
        NDArray
            Array of non-symmetric coordinates in this system's native coordinate system, shaped ``(..., NDIM_NSIM)``.

        Raises
        ------
        ValueError
            If the Cartesian coordinates cannot be converted.

        Notes
        -----
        The symmetry determines which axes are invariant. Only non-symmetric coordinates
        are included in the output.
        """
        if symmetry is None:
            symmetry = self.symmetry
        try:
            # Convert Cartesian to native coordinates
            native_coordinates = self.coordinate_system.from_cartesian(coordinates)
            # Return only non-symmetric coordinates
            return native_coordinates[..., ~symmetry._invariance_array]
        except Exception as e:
            raise ValueError(f"Failed to convert from Cartesian coordinates: {e}")

    def to_cartesian(self, coordinates: NDArray, symmetry: Symmetry=None) -> NDArray:
        """
        Convert coordinates from the current coordinate system to Cartesian coordinates.

        Parameters
        ----------
        coordinates : NDArray
            Coordinates in the current coordinate system, shaped ``(..., NDIM)``.
        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to the symmetry of this handler.

        Returns
        -------
        NDArray
            Cartesian coordinates corresponding to the input coordinates, shaped ``(..., 3)``.

        Raises
        ------
        ValueError
            If the input coordinates cannot be validated or converted.

        Notes
        -----
        Missing coordinates are filled with default values based on the specified or default symmetry.
        """
        if symmetry is None:
            symmetry = self.symmetry
        try:
            # Fill missing coordinates based on symmetry
            coordinates = self.fill_missing_coordinates(coordinates, symmetry=symmetry)
            # Convert native coordinates to Cartesian
            return self.coordinate_system.to_cartesian(coordinates)
        except Exception as e:
            raise ValueError(f"Failed to convert to Cartesian coordinates: {e}")

    def from_grid(self, coordinates: NDArray, axes: Optional[List[str]] = None, symmetry: Symmetry=None) -> NDArray:
        """
        Convert Cartesian grid coordinates to the native coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of Cartesian coordinates with shape ``(..., len(axes))``.
        axes : Optional[List[str]], optional
            List of axes corresponding to the coordinates provided. If None, assumes
            the first `len(coordinates.shape[-1])` axes of the Cartesian coordinate system
            (e.g., ['x', 'y', 'z'][:len(coordinates.shape[-1])]).
        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to the symmetry of this handler.

        Returns
        -------
        NDArray
            Array of coordinates in this system's native coordinate system, shaped ``(..., NDIM_NSIM)``.

        Raises
        ------
        ValueError
            If the Cartesian coordinates or axes are invalid.

        Notes
        -----
        This method converts Cartesian grid coordinates to native coordinates and filters
        out symmetric axes, returning only non-symmetric coordinates.
        """
        if symmetry is None:
            symmetry = self.symmetry
        try:
            # Convert grid coordinates to native system
            native_coordinates = self.coordinate_system.from_grid(coordinates, axes=axes)
            # Return only non-symmetric coordinates
            return native_coordinates[..., ~symmetry._invariance_array]
        except Exception as e:
            raise ValueError(f"Failed to convert from grid coordinates: {e}")

    def to_grid(self, coordinates: NDArray, axes: Optional[List[str]] = None, symmetry: Symmetry =None) -> NDArray:
        """
        Convert native coordinates in this coordinate system to a Cartesian grid with specified axes.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates in this system's native format, shaped ``(..., NDIM)``.
        axes : Optional[List[str]], optional
            List of axes for the output Cartesian grid. If None, assumes the first `NDIM` axes
            of the Cartesian coordinate system.
        symmetry : Optional[Symmetry], optional
            Symmetry object specifying invariant axes. Defaults to the symmetry of this handler.

        Returns
        -------
        NDArray
            Array of Cartesian coordinates in the specified axes, shaped ``(..., len(axes))``.

        Raises
        ------
        ValueError
            If the native coordinates or axes are invalid.

        Notes
        -----
        Missing coordinates are filled with default values based on the specified or default symmetry,
        and the output is filtered to include only the specified axes.
        """
        if symmetry is None:
            symmetry = self.symmetry
        try:
            # Fill missing coordinates based on symmetry
            coordinates = self.fill_missing_coordinates(coordinates, symmetry=symmetry)
            # Convert native coordinates to Cartesian grid coordinates
            return self.coordinate_system.to_grid(coordinates, axes=axes)
        except Exception as e:
            raise ValueError(f"Failed to convert to grid coordinates: {e}")

    def as_dict(self) -> dict:
        """
        Convert the GeometryHandler instance to a dictionary for serialization.

        Returns
        -------
        dict
            A dictionary representation of the GeometryHandler instance.

        Raises
        ------
        TypeError
            If `fill_values` cannot be converted to a list for JSON compatibility.
        """
        try:
            return {
                'coordinate_system': {
                    'class_name': self.coordinate_system.__class__.__name__,
                    'args': self.coordinate_system._args,
                    'kwargs': self.coordinate_system._kwargs
                },
                'symmetry': np.arange(self.coordinate_system.NDIM)[
                    self.symmetry._invariance_array].tolist() if self.symmetry else None,
                'fill_values': self.fill_values.tolist()  # Convert to list for JSON compatibility
            }
        except TypeError as e:
            raise TypeError(f"Failed to serialize fill_values: {e}")

    @classmethod
    def from_dict(cls, dct: dict[Any, Any]) -> 'GeometryHandler':
        """
        Create a GeometryHandler instance from a dictionary representation.

        Parameters
        ----------
        dct : dict
            A dictionary containing serialized GeometryHandler attributes.

        Returns
        -------
        GeometryHandler
            An instance of GeometryHandler initialized from the dictionary.

        Raises
        ------
        ValueError
            If required keys ('coordinate_system') are missing or improperly formatted in the dictionary.
        """
        if 'coordinate_system' not in dct:
            raise ValueError("Dictionary is missing required key 'coordinate_system'.")

        try:
            coord_class = find_in_subclasses(CoordinateSystem, dct['coordinate_system']['class_name'])
            coordinate_system = coord_class(*dct['coordinate_system']['args'], **dct['coordinate_system']['kwargs'])
        except Exception as e:
            raise ValueError(f"Failed to load coordinate system from dictionary: {e}")

        # Load symmetry
        try:
            symmetry = Symmetry(dct['symmetry'], coordinate_system) if dct['symmetry'] else None
        except Exception as e:
            raise ValueError(f"Failed to initialize symmetry: {e}")

        # Validate and load fill values
        fill_values = np.array(dct.get('fill_values', []), dtype=np.float64)
        if fill_values.size == 0:
            raise ValueError("Fill values are missing or empty in the provided dictionary.")

        return cls(coordinate_system=coordinate_system, symmetry=symmetry, fill_values=fill_values)

    def to_hdf5(self, handle: Union[h5py.File, h5py.Group]):
        """
        Save the GeometryHandler instance to an HDF5 file or group.

        Parameters
        ----------
        handle : Union[h5py.File, h5py.Group]
            The HDF5 file or group handle to save data to.

        Raises
        ------
        TypeError
            If the handle is not a valid HDF5 file or group.
        ValueError
            If saving symmetry or fill values fails.
        """
        if not isinstance(handle, (h5py.File, h5py.Group)):
            raise TypeError("Handle must be an h5py.File or h5py.Group instance.")

        # Save the coordinate system
        coord_system_handle = handle.require_group('COORD_SYSTEM')
        self.coordinate_system.to_file(coord_system_handle, fmt='hdf5')

        # Save symmetry and fill values as attributes
        if self.symmetry is not None:
            try:
                handle.attrs['symmetry'] = np.array(
                    np.arange(self.coordinate_system.NDIM)[self.symmetry._invariance_array])
            except Exception as e:
                raise ValueError(f"Failed to save symmetry: {e}")

        try:
            handle.attrs['fill_values'] = self.fill_values
        except Exception as e:
            raise ValueError(f"Failed to save fill_values: {e}")

    @classmethod
    def from_hdf5(cls, handle: Union['h5py.File', 'h5py.Group']) -> 'GeometryHandler':
        """
        Load a GeometryHandler instance from an HDF5 file or group.

        Parameters
        ----------
        handle : Union[h5py.File, h5py.Group]
            The HDF5 file or group handle to load data from.

        Returns
        -------
        GeometryHandler
            An instance of GeometryHandler initialized from the HDF5 data.

        Raises
        ------
        TypeError
            If the handle is not a valid HDF5 file or group.
        KeyError
            If required data is missing from the HDF5 file.
        """
        if not isinstance(handle, (h5py.File, h5py.Group)):
            raise TypeError("Handle must be an h5py.File or h5py.Group instance.")

        # Load the coordinate system
        try:
            coordinate_system = CoordinateSystem.from_file(handle['COORD_SYSTEM'], fmt='hdf5')
        except KeyError as e:
            raise KeyError(f"Failed to load coordinate system from HDF5: {e}")

        # Load symmetry and fill values
        try:
            symmetry_array = handle.attrs.get('symmetry', None)
            symmetry = Symmetry(symmetry_array, coordinate_system) if symmetry_array is not None else None
        except Exception as e:
            raise ValueError(f"Failed to load symmetry from HDF5: {e}")

        try:
            fill_values = handle.attrs.get('fill_values',None)
            if fill_values is None:
                raise ValueError("Fill values are missing or empty in the HDF5 file.")
            fill_values = np.array(fill_values, dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to load fill_values from HDF5: {e}")

        return cls(coordinate_system=coordinate_system, symmetry=symmetry, fill_values=fill_values)
