"""
Abstract base classes for Pisces geometry. In this module, the base class :py:class:`CoordinateSystem` is defined
and its corresponding metaclass is also defined.


Developer Notes
===============

This module defines abstract base classes for constructing orthogonal coordinate systems and symmetry representations
within the Pisces framework. These classes, :py:class:`CoordinateSystem` and :py:class:`Symmetry`, are intended as base classes for developers
to create custom coordinate systems and symmetries. Key elements include customizable Lame coefficient mappings,
dependency matrices, and robust data storage methods supporting JSON, YAML, and HDF5 formats.

Coordinate Systems
------------------

The :py:class:`CoordinateSystem` class defines a flexible framework for modeling orthogonal coordinate systems. It introduces
core attributes and methods to manage transformations, differential operations, and geometric properties specific
to each coordinate system. The structure is highly modular, allowing each coordinate axis to be associated with a unique
Lame coefficient function, thus supporting non-Cartesian geometries.

**Core Features**:

- Transformation rules to and from Cartesian coordinates.
- Lame Coefficients for differential operations.
- Dependence tracking for Lame Coefficients to ensure optimized differential operations.
- Serialization: to HDF5, YAML, and JSON.

Transformation Rules
++++++++++++++++++++

Every :py:class:`CoordinateSystem` class implements the :py:meth:`CoordinateSystem._convert_cartesian_to_native` and
:py:meth:`CoordinateSystem._convert_native_to_cartesian`. These are then ported through the :py:meth:`CoordinateSystem.to_cartesian` and
:py:meth:`CoordinateSystem.from_cartesian` methods (which should not be altered in subclasses) to provide core transformation rules.

To convert between two coordinate systems, one of two things is done. If there is a built-in method with signature ``.to_<coordinate_system_name>``
in the class, then a direct conversion link exists and is used. Otherwise, we convert to cartesian coordinates and then back to the
other coordinate system. This ensures that we can seamlessly convert between any set of coordinate systems.

.. note::

    Under the hood, the logic for seeking the special ``.to_<coordinate_system_name>`` is built into the :py:class:`geometry.utils.CoordinateConverter` class.
    When a conversion is created using :py:meth:`CoordinateSystem.convert_to`, it creates such a converter and returns it
    to handle the conversion on its own.

Lame Coefficients
+++++++++++++++++

In any :py:class:`CoordinateSystem`, Lame coefficients are essential scaling factors that adjust the differential
operators to account for the local geometry of each axis. These coefficients, represented as functions, adapt differential
operations to non-Cartesian coordinate systems. Each axis in an orthogonal coordinate system has a corresponding Lame
coefficient function, which typically varies with position in space. For example, in spherical coordinates, the Lame
coefficient for the radial coordinate is constant, while the coefficients for angular coordinates depend on the radial distance.

Every CoordinateSystem subclass defines Lame coefficient methods using the :py:func:`geometry.utils.lame_coefficient` decorator, specifying
which axis each function corresponds to. These methods are then automatically collected by the :py:class:`CoordinateSystemMeta`
metaclass. The metaclass processes these methods and maps each to its respective axis, ensuring that each coordinate axis
has a valid Lame coefficient function, which enables efficient and accurate differential calculations.

The metaclass also initializes the ``_lame_dependence_matrix``, which is a binary matrix indicating the dependencies between
Lame coefficients and coordinate axes. The method :py:meth:`CoordinateSystemMeta._solve_lame_dependence` computes this
matrix based on the Lame coefficient functions' specified dependencies. This matrix optimizes the computational load by
only including essential dependencies for each differential operation. For example, if a Lame coefficient depends on
a single axis, computations involving that coefficient will exclude any unrelated axes.

The :py:meth:`CoordinateSystem.compute_lame_coefficients` method retrieves the Lame coefficients for the specified coordinates
and axes, enabling selective computation when only certain coefficients are needed. By default, all coefficients are
computed, but users can specify active_axes to limit the calculations, which is particularly useful in symmetric coordinate systems.

Implementation Examples
++++++++++++++++++++++++

As a first example, this is the implementation for a 3D cartesian coordinate system

.. code-block::

    class CartesianCoordinateSystem(CoordinateSystem):

        NDIM = 3
        AXES = ['x', 'y', 'z']

        def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
            return coordinates  # Cartesian is already in native form

        def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
            return coordinates  # Cartesian is already in native form

        @lame_coefficient(0,axes=[])
        def lame_0(self,coordinates):
            return np.ones_like(coordinates[:,0])

        @lame_coefficient(1,axes=[])
        def lame_1(self,coordinates):
            return np.ones_like(coordinates[:,0])

        @lame_coefficient(2,axes=[])
        def lame_2(self,coordinates):
            return np.ones_like(coordinates[:,0])

Here, the Lame coefficients are all 1, which makes things very simple. In spherical coordinates we instead have

.. code-block:: python

    class SphericalCoordinateSystem(CoordinateSystem):

        NDIM = 3
        AXES = ['r', 'theta', 'phi']

        def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
            r, theta, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            return np.stack((x, y, z), axis=-1)

        def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
            x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(z / r)
            phi = np.arctan2(y, x)
            return np.stack((r, theta, phi), axis=-1)

        @lame_coefficient(0,axes=[])
        def lame_0(self,coordinates):
            return np.ones_like(coordinates[...,0])

        @lame_coefficient(1,axes=[0])
        def lame_1(self,coordinates):
            return coordinates[...,0]

        @lame_coefficient(2,axes=[0,1])
        def lame_2(self,coordinates):
            r, theta = coordinates[...,0],coordinates[...,1]
            return r*np.sin(theta)

"""
from abc import ABC, abstractmethod, ABCMeta
from typing import Any, List, Optional, Dict, TYPE_CHECKING, Collection, Callable

import numpy as np
from numpy.typing import NDArray

from pisces.geometry._exceptions import ConversionError
from pisces.geometry.utils import CoordinateConverter
from pisces.utilities.general import find_in_subclasses
from pisces.utilities.math import partial_derivative, function_partial_derivative

if TYPE_CHECKING:
    from pisces.geometry._typing import LameCoefficientMap

# noinspection PyProtectedMember,PyUnresolvedReferences
class CoordinateSystemMeta(ABCMeta):
    """
    Metaclass for CoordinateSystem that automatically gathers decorated Lame
    coefficient methods and creates a mapping of coefficients to their axes.

    This metaclass inspects all methods in a ``CoordinateSystem`` class, identifying
    those marked with the ``@lame_coefficient`` decorator. It then builds a mapping
    from each axis to the associated Lame coefficient function and validates that
    all necessary Lame coefficients are defined.

    Attributes
    ----------
    _cls_lame_map : Dict[int, str]
        A dictionary mapping each axis index to its Lame coefficient function.
    _lame_invariance_matrix : NDArray
        A matrix indicating dependencies of each Lame coefficient on specific coordinates. More precisely,
        the ``_lame_dependence_matrix`` is an ``(NDIM,NDIM)`` array such that the ``(i,j)`` element indicates if the
        :math:`i`-th Lame coefficient depends on the :math:`j`-th coordinate.

    Notes
    -----
    Lame coefficients :math:`h_i` define scaling factors for a given coordinate axis ``i`` in an
    orthogonal coordinate system. This metaclass allows subclasses of ``CoordinateSystem`` to
    define coordinate-specific Lame coefficient functions.
    """
    _cls_lame_map: Dict[int, str] = {}
    _lame_invariance_matrix: Optional[NDArray[bool]] = None

    def __init__(cls, name, bases, class_dict):
        # Initialize the standard object as normal.
        super().__init__(name, bases, class_dict)

        # ----------------------------------- #
        # Constructing Lame Coefficients      #
        # ----------------------------------- #
        # Here we start initializing the lame coefficients. We ensure that
        # they get registered and are accessible in the _lame_map and then
        # construct the invariance matrix for the coefficients.
        cls._cls_lame_map = {}
        cls._lame_invariance_matrix = None

        # LOCATING LAME COEFFICIENTS.
        # Iterate through all of the class attributes to find objects with
        # the lame marker and register them.
        for attr_name, attr_value in class_dict.items():
            if callable(attr_value) and getattr(attr_value, '_is_lame', False):
                # We've identified a Lame Coefficient function, we now need to register its axis and set the invariance
                # array for the method.
                axis = attr_value._lame_axis  # The axis this Lame coefficient corresponds to.
                required_axes = getattr(attr_value, 'required_axes', 'all')

                if required_axes == 'all':
                    required_axes = [i for i in range(class_dict['NDIM'])]

                attr_value.invariance = np.array([i not in required_axes for i in range(class_dict['NDIM'])],dtype=bool)

                del attr_value._lame_axis
                del attr_value.required_axes

                cls._cls_lame_map[axis] = attr_name

        # VALIDATE LAME COEFFICIENTS
        # If this isn't an abstract base class, we need to enforce that all of the
        # Lame Coefficients are provided.
        if not hasattr(cls, '__abstractmethods__') or not cls.__abstractmethods__:
            required_axes = range(class_dict['NDIM'])
            missing_axes = [axis for axis in required_axes if axis not in cls._cls_lame_map]

            if missing_axes:
                raise ValueError(
                    f"Missing Lame coefficient functions for axes: {missing_axes}. "
                    f"Each axis from 0 to {class_dict['NDIM'] - 1} must have a corresponding Lame coefficient."
                )

        # COMPUTE THE LAME DEPENDENCE MATRIX
        cls._lame_invariance_matrix = cls._solve_lame_dependence(name, bases, class_dict)

    def _solve_lame_dependence(cls, _, __, class_dict) -> np.ndarray:
        """
        Solve the dependency matrix for Lame coefficients, indicating if each Lame coefficient
        depends on specific coordinate axes.

        Returns
        -------
        np.ndarray
            A matrix of shape ``(NDIM, NDIM)`` where ``matrix[i, j] = 1`` if the Lame coefficient
            for axis ``i`` depends on the ``j``-th coordinate axis, and ``0`` otherwise.

        Notes
        -----
        This really relies in the decorator used to mark the Lame coefficient functions. By default, the dependence
        axes are ``"all"``, but they can (and should) be set in subclasses to ensure efficient runtime performance.
        """
        matrix = np.ones((class_dict['NDIM'], class_dict['NDIM']), dtype=bool)

        for i, lame_func in cls._cls_lame_map.items():
            # Get the required axes for this Lame coefficient function
            matrix[i, :] = class_dict[lame_func].invariance

        return matrix
class CoordinateSystem(ABC, metaclass=CoordinateSystemMeta):
    """
    Abstract base class representation of an orthogonal coordinate system.

    This class serves as a base for specific orthogonal coordinate systems, defining key
    mathematical methods to compute differential operators, transformations, and basis
    functions.

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
    NDIM: int = 3
    AXES: list[str] = ['x', 'y', 'z']

    def __init__(self, *args, **kwargs):
        # Setup the basic arguments and kwargs.
        self._args, self._kwargs = args, kwargs

        # Initialize the Lame coefficients.
        self._init_lame()

    # noinspection PyProtectedMember
    def _init_lame(self):
        self._lame_map = {}
        for k, v in self.__class__._cls_lame_map.items():
            self._lame_map[k] = getattr(self, v)

    def __repr__(self):
        """
        Return a string representation of the CoordinateSystem instance.

        Provides a concise summary, including the class name and dimensionality.

        Returns
        -------
        str
            A string representing the instance.
        """
        _args = ", ".join(self._args)
        _kwargs = ", ".join([f"{k}={v}" for k, v in self._kwargs])

        return f"{self.__class__.__name__}({_args},{_kwargs})"

    def __str__(self):
        _param_str = ""
        if len(self._args):
            _param_str += ", ".join(self._args)
        if len(self._kwargs):
            _param_str += ", ".join([f"{k}={v}" for k, v in self._kwargs])

        return f"{self.__class__.__name__}({_param_str})"

    def __eq__(self, other):
        # Check that the two are both coordinate systems and that they are the
        # same type of coordinate system.
        if isinstance(other, CoordinateSystem):
            if type(self) != type(other):
                return False
        else:
            raise TypeError(f"Cannot check for equality between {type(self)} and {type(other)}.")

        # Now check for equivalent args and kwargs.
        if len(self._args) != len(other._args):
            return False
        else:
            _s = all([sarg == oarg for sarg, oarg in zip(self._args, other._args)])
            if not _s:
                return False

        # Now check the kwargs.
        for k, v in self._kwargs.items():
            if k not in other._kwargs:
                return False
            if v != other._kwargs[k]:
                return False

        return True

    def __getitem__(self, item: int) -> str:
        try:
            return self.AXES[item]
        except KeyError:
            raise ValueError(f"No axis: {item}.")

    def __len__(self):
        return self.NDIM

    @property
    def lame_coefficients(self) -> 'LameCoefficientMap':
        r"""
        Returns a dictionary mapping each axis to its corresponding Lame coefficient function.

        Returns
        -------
        LameCoefficientMap
            A mapping of axes to their respective Lame coefficient functions.

        Notes
        -----
        In any orthogonal coordinate system, there is a set of *orthogonal* vectors :math:`{\bf e}_0, \cdots {\bf e}_N` such that,
        at any point in the space, the :math:`i`-th basis vector points in the direction of increase for the corresponding coordinate
        component. By definition, we call this set of basis vectors the covariant basis and define it as

        .. math::

            {\bf e}_i = \frac{\partial {\bf r}}{\partial q^i},

        where :math:`{\bf r}` is the position in space and :math:`q^i` are the coordinates.

        .. hint::

            In cartesian coordinates, these are constant everywhere in space. That is not true in a general orthog. coord.
            system.

        These basis vectors are a self-consistent basis; however, they may or may not be unit vectors. Thus, we define
        the unit basis as

        .. math::

            \hat{\bf e}_i = \frac{{\bf e}_i}{|{\bf e}_i|}.

        This scaling factor is called the **Lame Coefficient** for that axis. It is generically a function of the entire space.

        This method returns the Lame coefficient functions for each axis as functions of the space.

        Examples
        --------

        To compute the Lame coefficients along the first axis of a set of points, we can proceed as follows:

        .. code-block:: python

            # Import the coordinate system you intend to use.
            from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
            import numpy as np

            # Initialize the coordinate system. This is necessary because other
            # (more complex) coordinate systems have arguments that would be
            # passed here.
            cs = SphericalCoordinateSystem()

            # Get your array of points.
            # In this case, a 100x100 grid of phi and theta.
            phi,theta = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
            r = np.ones_like(phi)

            coordinates = np.stack([r.ravel(),theta.ravel(),phi.ravel()],axis=1)

            # Now we can compute the lame coefficients.
            lame_r = cs.lame_coefficients[0](coordinates)

        For systems like cylindrical coordinates, the Lame coefficients for the angular axis differ based on radius.
        Here’s how to compute these in a custom cylindrical system:

        .. code-block:: python

            from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
            import numpy as np

            cs = CylindricalCoordinateSystem()
            coordinates = np.array([[1.0, np.pi/4, 2.0], [2.0, np.pi/2, 3.0]])  # (r, theta, z)

            # Compute the Lame coefficient for theta.
            lame_theta = cs.lame_coefficients[1](coordinates)
            print("Lame coefficient for theta:", lame_theta)
        """
        return self._lame_map

    # noinspection PyProtectedMember
    @property
    def lame_invariance_matrix(self) -> NDArray[bool]:
        r"""
        Returns a matrix indicating dependencies of each Lame coefficient on specific coordinates.

        Returns
        -------
        NDArray[bool]
            Matrix where ``matrix[i, j] = 1`` if the Lame coefficient for axis ``i`` depends on axis ``j``, and ``0`` otherwise.

        Notes
        -----

        Let :math:`q^1,\cdots,q^N` be an :math:`N` dimension coordinate system. Let :math:`\phi: \mathbb{R}^N \to \mathbb{R}`
        be a scalar field over the corresponding space. Generically, the coordinates may be partitioned into two groups, the
        symmetric coordinates and the non-symmetric coordinates. Formally,

        .. math::

            Q(\phi) = \left\{q^k \;\forall k < N \in \mathbb{N} \mid \partial_k \phi_{\bf r} = 0\;\forall {\bf r} \in \mathbb{R}^N\right\}

        is the **symmetric axis set** of :math:`\phi`. The **Lame invariance matrix**, is defined such that

        .. math::

            {\rm LIM}_{ij} = \begin{cases}1,&q^i \in Q(\lambda_j)\\0,&\text{otherwise.}\end{cases}

        The LIM is useful for determining which terms in a differential expression should include a particular Lame coefficient.
        For example, in spherical coordinates, the radial Lame coefficient depends only on the radial distance, while the angular
        coefficients depend on both radial distance and angle, so the LIM helps exclude redundant terms in complex calculations.

        Example
        -------
        To see the dependency structure of the Lame coefficients for a coordinate system, we can use a custom :py:class:`CoordinateSystem`
        class, say :py:class:`~geometry.coordinate_systems.SphericalCoordinateSystem`:

        .. code-block:: python

            import numpy as np
            from pisces.geometry.coordinate_systems import SphericalCoordinateSystem

            # Instantiate the coordinate system
            cs = SphericalCoordinateSystem()

            # Retrieve the Lame invariance matrix
            lame_invariance_matrix = cs.lame_invariance_matrix
            print("Lame invariance matrix:")
            print(lame_invariance_matrix)

        This matrix reveals which Lame coefficients depend on specific coordinates for the :py:class:`~geometry.coordinate_systems.SphericalCoordinateSystem`.
        For instance, a 3x3 matrix for spherical coordinates would show ``True`` values in entries indicating dependencies (e.g., radial distance
        might only impact angular coefficients), simplifying calculations by focusing on dependent coordinates only.

        See Also
        --------
        CoordinateSystem.lame_coefficients : Returns the Lame coefficients themselves, which use this invariance matrix.
        CoordinateSystem.compute_gradient_term : Demonstrates how to use the Lame invariance matrix to simplify calculations.
        """
        return self.__class__._lame_invariance_matrix

    def compute_lame_coefficients(
            self,
            coordinates: NDArray,
            active_axes: Optional[Collection[int]] = None
    ) -> NDArray:
        """
        Compute the Lame coefficients at specified coordinates for selective or all axes.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(..., NDIM)``.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the calculation. By default, all axes are computed.
            Limiting to specific axes can improve efficiency, especially in systems with symmetries
            that simplify computation by ignoring invariant coordinates.

        Returns
        -------
        NDArray
            Array of Lame coefficients at each specified coordinate, shaped ``(..., NDIM)`` by default.
            If ``active_axes`` is set, the shape is ``(..., len(active_axes))``.

        Notes
        -----
        Lame coefficients, :math:`h_i`, are critical in orthogonal coordinate systems. They scale
        differential operators based on local geometry and vary depending on the coordinate system.
        Specifying ``active_axes`` helps optimize calculations by excluding symmetrically invariant axes.

        .. warning::

            When using ``active_axes``, the result may not align with the axis indices directly.
            Ensure that the returned array indices are matched to ``active_axes`` rather than the
            full coordinate axes.

        Examples
        --------
        Calculate Lame coefficients at specific points in a custom coordinate system:

        .. code-block:: python

            # Import the relevant coordinate system.
            from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
            import numpy as np

            # Construct the coordinate array.
            x = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            cs = SphericalCoordinateSystem()
            lame_coeffs = cs.compute_lame_coefficients(x, active_axes=[0, 2])
            print("Lame Coefficients for axes 0 and 2:", lame_coeffs)

        See Also
        --------
        :py:meth:`lame_invariance_matrix` : Indicates dependencies between Lame coefficients and coordinate axes.
        """
        # Default to all axes if active_axes is None
        if active_axes is None:
            active_axes = range(self.NDIM)

        # Use list comprehension to build the array, one column per active axis
        result_array = np.column_stack(
            [self.lame_coefficients[axis](coordinates) for axis in active_axes]
        )

        return result_array

    def jacobian(self, coordinates):
        """
        Compute the Jacobian determinant for the given coordinate system at specified points.

        The Jacobian is calculated as the product of the Lame coefficients across all axes at each
        point in the coordinate system, representing the volume scaling factor for the transformation
        from Cartesian coordinates to the current coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shape ``(P, N)``, where ``P`` is the number of points and ``N``
            is the number of dimensions in the coordinate system.

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

        Examples
        --------
        Calculate the Jacobian determinant at several points in a custom coordinate system:

        >>> coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> cs = CoordinateSystem()  # Assume CoordinateSystem is defined with Lame coefficients
        >>> jacobian = cs.jacobian(coords)
        >>> print("Jacobian determinant:", jacobian)

        """
        return np.prod(self.compute_lame_coefficients(coordinates), axis=0)

    def effective_jacobian(self, coordinates, axis: int):
        r"""
        Compute the effective Jacobian determinant for the given coordinate system at specified points along a given axis.

        The Jacobian is calculated as the product of the Lame coefficients across all axes at each
        point in the coordinate system, representing the volume scaling factor for the transformation
        from Cartesian coordinates to the current coordinate system. The effective Jacobian is constructed by
        removing terms from the product which do not depend on a particular axis. It is useful primarily in calculations
        of the divergence, curl, and Laplacian.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shape ``(P, N)``, where ``P`` is the number of points and ``N``
            is the number of dimensions in the coordinate system.
        axis: int
            The axis for which to construct the effective Jacobian determinant.

        Returns
        -------
        NDArray
            The Jacobian determinant values at each specified point, with shape ``(P,)``. Each
            value represents the volume scaling factor at that point.

        Notes
        -----
        Generically, the Jacobian determinant is constructed as the product of the Lame Coefficients:

        .. math::

            J = \prod_i \lambda_i.

        In many calculations in geometry, the Jacobian appears. For example, the divergence is

        .. math::

            \nabla \cdot {\bf F} = \frac{1}{J}\sum_i \partial_i \left(J \hat{\bf F}_i\right).

        Clearly, the components of the Jacobian product that are invariant under :math:`q^i` will actually cancel in
        such an expression. Thus, we introduce the notion of the effective Jacobian determinant:

        .. math::

            \tilde{J}_k = \prod_{i \in \{j|\lambda_k \notin {\rm Inv}_{q^j}\}} \lambda_i.

        Then,

        .. math::

            \nabla \cdot {\bf F} = \sum_i \frac{1}{\tilde{J}_i} \partial_i \left(\tilde{J}_i \hat{\bf F}_i\right).

        From a numerical standpoint, this is a considerable improvement as it ensures that we are not forced to compute
        derivatives on a (potentially complicated) Jacobian just to have them from back out of the equation again.

        Examples
        --------
        Calculate the Jacobian determinant at several points in a custom coordinate system:

        >>> coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> cs = CoordinateSystem()  # Assume CoordinateSystem is defined with Lame coefficients
        >>> jacobian = cs.effective_jacobian(coords)
        >>> print("Jacobian determinant:", jacobian)

        """
        active_axes = np.arange(self.NDIM)[~self.lame_invariance_matrix[:, axis]]
        return np.prod(self.compute_lame_coefficients(coordinates, active_axes=active_axes), axis=0)

    def surface_element(self, coordinates: NDArray, axis: int) -> NDArray:
        """
        Compute the surface element along a specified axis for the given coordinate system at specified points.

        The surface element represents the area scaling factor for a surface normal to the specified axis, computed
        as the product of the Lame coefficients for all other axes at each point in the coordinate system. This is
        useful for integrations over surfaces in non-Cartesian coordinate systems.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates, shape ``(P, N)``, where ``P`` is the number of points and ``N``
            is the number of dimensions in the coordinate system.
        axis : int
            The index of the axis normal to the surface for which to compute the surface element.
            This axis will be excluded from the computation, and the surface element will be calculated
            from the Lame coefficients of all other axes.

        Returns
        -------
        NDArray
            The surface element values at each specified point, with shape ``(P,)``. Each
            value represents the area scaling factor for the surface normal to the specified axis
            at that point.

        Notes
        -----
        In non-Cartesian coordinate systems, the surface element is the product of the Lame
        coefficients for all axes except the specified axis. This can be expressed as:

        .. math::
           dS = h_1 h_2 \dots h_{N-1}

        where :math:`h_i` are the Lame coefficients for each axis excluding the specified axis.
        """
        _lame_coefficients = self.compute_lame_coefficients(coordinates)
        _lame_coefficients = np.delete(_lame_coefficients, axis, axis=0)
        return np.prod(_lame_coefficients, axis=0)

    def compute_gradient_term(self,
                              coordinates: NDArray,
                              values: NDArray,
                              axis: int,
                              /,
                              derivative: NDArray = None,
                              *,
                              basis='unit',
                              **kwargs):
        r"""
        Compute a single term of the gradient for a scalar field with respect to a specified axis.

        This function calculates the partial derivative of a scalar field, ``values``, along a
        specified axis in a coordinate system. The result is adjusted based on the provided ``basis`` parameter,
        scaling the derivative with the appropriate Lame coefficient if necessary.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points on a grid. This should be ``(*GRID,NDIM) = (N_0,...,N_NDIM,NDIM)`` in shape, where the first
            ``NDIM`` dimensions in the array correspond to the number of grid points along that axis.
        values : NDArray
            Array of scalar field values at the specified coordinates, shaped ``(*GRID,)`` or ``(*GRID,1)``.
        axis : int
            Index of the axis along which to compute the gradient term.
        derivative : Optional[NDArray], optional
            Precomputed partial derivative along the specified axis, shaped ``(*GRID,)`` or ``(*GRID,1)``.
            If not provided, it will be computed within the function.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which to compute the gradient term. Default is ``'unit'``:
            - ``'unit'``: Returns derivatives scaled by the Lame coefficient.
            - ``'covariant'``: Returns derivatives scaled by the square of the Lame coefficient.
            - ``'contravariant'``: Returns unscaled derivatives, ignoring Lame coefficients.
        **kwargs
            Additional keyword arguments passed to the ``partial_derivative`` function (if derivative is not provided),
            such as ``edge_order``.

        Returns
        -------
        NDArray
            Computed gradient term along the specified axis, adjusted based on the selected basis. Shaped
            ``(*GRID,)``.

        Notes
        -----
        The choice of ``basis`` affects the scaling of the gradient term:

        - **Unit basis**: Scales the partial derivative by the Lame coefficient, :math:`h_i`.
        - **Covariant basis**: Scales by the square of the Lame coefficient, :math:`h_i^2`.
        - **Contravariant basis**: Direct partial derivative, no scaling by Lame coefficient.

        In non-Cartesian coordinate systems, this distinction is crucial for accurate computation.
        If no precomputed derivative is provided, ``partial_derivative`` will be invoked with the
        specified axis to compute it.

        See Also
        --------
        partial_derivative : Used for computing derivatives along specified axes.
        gradient : Computes the full gradient vector for a scalar field across all axes.
        """
        # Compute the relevant partial derivative if that is necessary.
        if derivative is None:
            derivative = partial_derivative(coordinates, values,axes=[axis], **kwargs) # S = (...,)

        if basis == 'contravariant':
            # In the contravariant basis, the lame coefficients don't appear and therefore
            # we don't need to compute them. We just return the derivatives.
            return derivative

        # Compute the lame coefficients.
        # shape = (N,1)
        lame_coefficients = self.lame_coefficients[axis](coordinates)

        # Return the other relevant basis options.
        if basis == 'covariant':
            # scaled by the square lame coefficients.
            return derivative / lame_coefficients ** 2
        elif basis == 'unit':
            return derivative / lame_coefficients
        else:
            raise ValueError(
                f"Unrecognized value for 'basis': {basis}. Expected one of 'unit', 'covariant', 'contravariant'.")

    def compute_function_gradient_term(self,
                                       coordinates: NDArray,
                                       function: Callable[[NDArray], NDArray],
                                       axis: int,
                                       *,
                                       derivative: Optional[Callable[[NDArray], NDArray]] = None,
                                       basis='unit',
                                       epsilon: float = 1e-5) -> NDArray:
        r"""
        Compute a single term of the gradient for a scalar function with respect to a specified axis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points on a grid.
        function : Callable[[NDArray], NDArray]
            Scalar function to be differentiated.
        axis : int
            Index of the axis for gradient computation.
        derivative : Optional[Callable[[NDArray], NDArray]], optional
            A precomputed derivative function for the specified axis. If None, it defaults to numerical differentiation.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which to compute the gradient term.
        epsilon : float, optional
            Step size for numerical differentiation.

        Returns
        -------
        NDArray
            Computed gradient term along the specified axis, adjusted based on the selected basis.
        """
        if derivative is None:
            derivative_result = function_partial_derivative(function, coordinates, axis, method='central',
                                                           h=epsilon)
        else:
            derivative_result = derivative(coordinates)

        lame_coefficients = self.lame_coefficients[axis](coordinates)

        if basis == 'unit':
            return derivative_result / lame_coefficients
        elif basis == 'covariant':
            return derivative_result / lame_coefficients ** 2
        elif basis == 'contravariant':
            return derivative_result
        else:
            raise ValueError(f"Unrecognized basis '{basis}'. Expected one of 'unit', 'covariant', or 'contravariant'.")

    def gradient(self,
                 coordinates: NDArray,
                 values: NDArray,
                 /,
                 derivatives: NDArray = None,
                 *,
                 basis='unit',
                 active_axes: List[int] | None = None,
                 **kwargs):
        """
        Compute the gradient of a scalar field with respect to coordinates in a specified basis.

        This method calculates the gradient of ``values`` (a scalar field) with respect to ``coordinates``
        in a given coordinate system, accounting for different bases: unit, covariant, or contravariant.
        The gradient is scaled by the appropriate Lame coefficients if required by the basis.

        Parameters
        ----------
        coordinates : NDArray
            Grid of coordinates with generic shape ``(...,NDIM)``, where the first ``NDIM`` axes correspond to the
            grid structure and the final axis corresponds to each of the coordinate values.
        values : NDArray
            Array of scalar field values at the specified coordinates, with shape ``(...,)`` or ``(..., 1)``.
        derivatives : Optional[NDArray], optional
            Array of precomputed partial derivatives along all axes, with shape ``(..., NDIM)``. If ``None``,
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
        **kwargs
            Additional keyword arguments passed to ``partial_derivatives_all_axes``, such as ``edge_order``.

        Returns
        -------
        NDArray
            Array representing the gradient of ``values`` with respect to ``coordinates``, shaped ``(..., NDIM)``.
            Each row represents the partial derivative along one axis, scaled according to the specified basis.
            If ``active_axes`` is specified, then the return shape will be ``(...,len(active_axes))``.

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
        # Construct the active axes so that, if they are not provided, we use
        # all of the axes and get all of the components back.
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        # Compute the gradients term by term.
        if derivatives is not None:
            return np.stack([
                self.compute_gradient_term(coordinates, values, axis, derivative=derivatives.take(axis,axis=-1), basis=basis,
                                           **kwargs) for axis in active_axes
            ], axis=1)
        else:
            return np.stack([
                self.compute_gradient_term(coordinates, values, axis, derivative=None, basis=basis, **kwargs) for axis
                in active_axes
            ], axis=1)

    def function_gradient(self,
                          coordinates: NDArray,
                          function: Callable[[NDArray], NDArray],
                          *,
                          derivatives: Optional[List[Callable[[NDArray], NDArray]]] = None,
                          basis='unit',
                          active_axes: Optional[List[int]] = None,
                          epsilon: float = 1e-5) -> NDArray:
        """
        Compute the gradient of a scalar function with respect to coordinates in a specified basis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points.
        function : Callable[[NDArray], NDArray]
            Scalar function to compute the gradient for.
        derivatives : Optional[List[Callable[[NDArray], NDArray]]], optional
            A list of derivative functions for each axis. If None, numerical derivatives are used.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Basis for computing the gradient, default is 'unit'.
        active_axes : List[int], optional
            List of active axis indices to compute the gradient.
        epsilon : float, optional
            Step size for finite differences.

        Returns
        -------
        NDArray
            Gradient of the function in the specified basis.
        """
        if active_axes is None:
            active_axes: List[int] = list(np.arange(self.NDIM))

        gradients = []
        for i, axis in enumerate(active_axes):
            derivative_func = derivatives[i] if derivatives is not None else None
            gradient_term = self.compute_function_gradient_term(
                coordinates, function, axis=axis, derivative=derivative_func, basis=basis, epsilon=epsilon
            )
            gradients.append(gradient_term)

        return np.stack(gradients, axis=-1)

    def compute_divergence_term(self, coordinates: NDArray,
                                vector_field: NDArray,
                                axis: int,
                                /,
                                basis='unit',
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
            Array of coordinate points with shape ``(..., NDIM)``, where ``...`` is the number of points
            and ``NDIM`` is the number of dimensions.
        vector_field : NDArray
            Array of vector field components at each coordinate point, with shape ``(..., NDIM)``.
        axis : int
            The index of the axis along which the divergence term is computed.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which *the vector field is provided*. Defaults to ``'unit'``. This will change
            the equation by a scaling factor (the Lame coefficient).

            .. hint::

                In most cases, we express vector fields in the ``"unit"`` basis, that way the vectors don't
                implicitly scale at different points in space (as they would if covariant or contravariant); however,
                there may be some instances in which this kwarg must be set.

        **kwargs
            Additional arguments to be passed to ``partial_derivative``, such as ``edge_order``.

        Returns
        -------
        NDArray
            The computed divergence term along the specified axis, with shape ``(...,)``.

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
        _vf = vector_field[:, axis]

        if basis == 'unit':
            _vf /= self.lame_coefficients[axis](coordinates)
        if basis == 'covariant':
            _vf /= self.lame_coefficients[axis](coordinates) ** 2
        elif basis == 'contravariant':
            pass
        else:
            raise ValueError(f"Unsupported basis '{basis}'. Expected 'unit' or 'contravariant'.")

        # Determine the Lame coefficient dependence for this axis. If a Lame coefficient doesn't depend
        # on a particular axis, then it pulls out of the derivative and cancels out of the Jacobian.
        dependent_axes = np.arange(self.NDIM)[~self.lame_invariance_matrix[axis, :]]

        # Compute the dependent part of the jacobian. This is the terms in the Jacobian that don't cancel
        # out.
        if len(dependent_axes):
            dependent_jacobian = np.prod(self.compute_lame_coefficients(coordinates, active_axes=dependent_axes),
                                         axis=0)
        else:
            # If we have no dependent axes, then this becomes trivial.
            dependent_jacobian = np.ones((coordinates.shape[0],))

        # Compute the relevant derivative term. This is partial J*F, where J is the dependent part
        # of the Jacobian. If the Jacobian is empty, then we just take the derivative of F.
        derivative_term = partial_derivative(coordinates, dependent_jacobian * _vf, axis, **kwargs)

        return derivative_term / dependent_jacobian

    def compute_function_divergence_term(self,
                                         coordinates: NDArray,
                                         vector_function: Callable[[NDArray], NDArray],
                                         axis: int,
                                         *,
                                         basis='unit',
                                         epsilon: float = 1e-5) -> NDArray:
        r"""
        Compute a single term of the divergence for an analytical vector function along a specified axis.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(..., NDIM)``, where ``...`` is the number of points.
        vector_function : Callable[[NDArray], NDArray]
            Analytical vector function to compute the divergence term for.
        axis : int
            The index of the axis along which the divergence term is computed.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which the vector function is provided. Defaults to 'unit'.
        epsilon : float, optional
            Step size for finite differences, used in numerical differentiation.

        Returns
        -------
        NDArray
            The computed divergence term along the specified axis, with shape ``(...,)``.

        Notes
        -----
        This function composes the vector function with the Lame coefficients before differentiating,
        enabling efficient computation of divergence terms with basis-specific scaling.
        """
        # Define the basis-corrected function
        lame_coeff = self.lame_coefficients[axis]
        if basis == 'unit':
            corrected_function = lambda coords: vector_function(coords)[:, axis] / lame_coeff(coords)
        elif basis == 'covariant':
            corrected_function = lambda coords: vector_function(coords)[:, axis] / (lame_coeff(coords) ** 2)
        elif basis == 'contravariant':
            corrected_function = lambda coords: vector_function(coords)[:, axis]
        else:
            raise ValueError(f"Unsupported basis '{basis}'. Expected 'unit', 'covariant', or 'contravariant'.")

        dependent_axes = np.arange(self.NDIM)[~self.lame_invariance_matrix[axis, :]]

        if len(dependent_axes):
            jacobian_component = lambda coords: np.prod(
                self.compute_lame_coefficients(coords, active_axes=dependent_axes), axis=0)
            composite_function = lambda coords: jacobian_component(coords) * corrected_function(coords)
        else:
            composite_function = corrected_function

        derivative_term = function_partial_derivative(composite_function, coordinates, axis, method='central', h=epsilon)

        if len(dependent_axes):
            # noinspection PyUnboundLocalVariable
            return derivative_term / jacobian_component(coordinates)
        else:
            return derivative_term

    def divergence(
            self,
            coordinates: NDArray,
            vector_field: NDArray,
            /,
            basis: str = 'unit',
            active_axes: Optional[List[int]] = None,
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
            Array of coordinate points with shape ``(..., NDIM)``, where ``...`` is the grid of points
            and ``NDIM`` is the number of dimensions.
        vector_field : NDArray
            Array of vector field components along each axis at each point, with shape ``(..., NDIM)``.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which *the vector field is provided*. Defaults to ``'unit'``:
            - ``'unit'``: Scales the vector field to avoid implicit scaling due to space curvature.
            - ``'covariant'``: Scales by :math:`h_i^2` for each component.
            - ``'contravariant'``: No scaling, assumes direct contravariant basis.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.

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
        # Validate inputs to ensure coordinates and vector field have compatible shapes
        if coordinates.shape != vector_field.shape:
            raise ValueError(
                f"Coordinates and vector_field have mismatched shapes: {coordinates.shape}, {vector_field.shape}")

        # Default to all axes if active_axes is not specified
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        # Initialize an array for the divergence result
        divergence_result = np.zeros(coordinates.shape[0])

        # Sum the divergence terms for each axis
        for axis in active_axes:
            divergence_result += self.compute_divergence_term(
                coordinates,
                vector_field,
                axis,
                basis=basis,
                **kwargs
            )

        return divergence_result

    def laplacian(
            self,
            coordinates: NDArray,
            scalar_field: NDArray,
            /,
            active_axes: Optional[List[int]] = None,
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
            Array of coordinate points with shape ``(..., NDIM)``, where ``...`` is the grid of points
            and ``NDIM`` is the number of dimensions.
        scalar_field : NDArray
            Array of scalar field values at each coordinate point, with shape ``(...,)``.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the Laplacian calculation. By default, all axes are included.
        kwargs :
            Additional kwargs to pass to :py:meth:`CoordinateSystem.gradient` and to :py:meth:`CoordinateSystem.divergence`.

        Returns
        -------
        NDArray
            The Laplacian of the scalar field at each coordinate point, with shape ``(...,)``.

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
        # Default to all axes if active_axes is not specified
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        # Compute the gradient in the unit basis, since divergence requires it in this form
        gradient_values = self.gradient(coordinates, scalar_field, basis='contravariant', active_axes=active_axes,
                                        **kwargs)

        # Compute the divergence of the gradient to obtain the Laplacian
        laplacian_values = self.divergence(coordinates, gradient_values, basis='contravariant', active_axes=active_axes,
                                           **kwargs)

        return laplacian_values

    def function_divergence(self,
                            coordinates: NDArray,
                            vector_function: Callable[[NDArray], NDArray],
                            *,
                            basis: str = 'unit',
                            active_axes: Optional[List[int]] = None,
                            epsilon: float = 1e-5) -> NDArray:
        """
        Compute the divergence of a vector function in this coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(..., NDIM)``.
        vector_function : Callable[[NDArray], NDArray]
            Function representing the vector field, returning an array of components at each point.
        basis : {'unit', 'covariant', 'contravariant'}, optional
            Specifies the basis in which the vector function is provided. Defaults to 'unit'.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.
        epsilon : float, optional
            Step size for finite differences, used in numerical differentiation.

        Returns
        -------
        NDArray
            The divergence of the vector field at each coordinate point, with shape ``(N,)``.
        """
        # Default to all axes if active_axes is not specified
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        # Initialize an array for the divergence result
        divergence_result = np.zeros(coordinates.shape[0])

        # Sum the divergence terms for each axis
        for axis in active_axes:
            divergence_result += self.compute_function_divergence_term(
                coordinates,
                vector_function,
                axis,
                basis=basis,
                epsilon=epsilon
            )

        return divergence_result

    def function_laplacian(self,
                                   coordinates: NDArray,
                                   scalar_function: Callable[[NDArray], NDArray],
                                   *,
                                   active_axes: Optional[List[int]] = None,
                                   epsilon: float = 1e-5) -> NDArray:
        """
        Compute the Laplacian of a scalar function in this coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinate points with shape ``(..., NDIM)``.
        scalar_function : Callable[[NDArray], NDArray]
            Function representing the scalar field, returning values at each point.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the Laplacian calculation. By default, all axes are included.
        epsilon : float, optional
            Step size for finite differences, used in numerical differentiation.

        Returns
        -------
        NDArray
            The Laplacian of the scalar field at each coordinate point, with shape ``(...,)``.
        """
        # Default to all axes if active_axes is not specified
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        # Define a gradient function to calculate the divergence of this gradient
        gradient_function = lambda coords: np.stack([
            self.compute_function_gradient_term(
                coords,
                scalar_function,
                axis,
                basis='contravariant',
                epsilon=epsilon
            ) for axis in active_axes
        ], axis=-1)

        # Compute the divergence of the gradient to obtain the Laplacian
        laplacian_values = self.function_divergence(
            coordinates,
            gradient_function,
            basis='contravariant',
            active_axes=active_axes,
            epsilon=epsilon
        )

        return laplacian_values

    def to_cartesian(self, coordinates) -> NDArray:
        """
        Convert native coordinates of this coordinate system to Cartesian coordinates.

        This method transforms coordinates from the system defined by this instance to the standard
        Cartesian system (e.g., [x, y, z] in 3D space). The conversion function ``_convert_native_to_cartesian``
        must be implemented by subclasses for this operation.

        Parameters
        ----------
        coordinates : NDArray
            Array of native coordinates in this coordinate system, with shape ``(..., NDIM)``, where ``...`` is
            the grid of points and ``NDIM`` is the number of dimensions.

        Returns
        -------
        NDArray
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
        try:
            return self._convert_native_to_cartesian(coordinates)
        except Exception as e:
            raise ConversionError(f"Failed to convert from {self.__class__.__name__} to cartesian: {e}")

    def from_cartesian(self, coordinates) -> NDArray:
        """
        Convert Cartesian coordinates to the native coordinates of this coordinate system.

        This method converts points from the Cartesian coordinate system into the coordinates
        specific to this coordinate system instance. The conversion function ``_convert_cartesian_to_native``
        must be implemented by subclasses for this operation.

        Parameters
        ----------
        coordinates : NDArray
            Array of Cartesian coordinates with shape ``(..., NDIM)``, where ``...`` may be any grid structure and
            ``NDIM`` is the number of dimensions.

        Returns
        -------
        NDArray
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
        try:
            return self._convert_cartesian_to_native(coordinates)
        except Exception as e:
            raise ConversionError(f"Failed to convert from cartesian to  {self.__class__.__name__}: {e}")

    def convert_to(self, target_coord_system: 'CoordinateSystem', *args: Any) -> NDArray:
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
        NDArray
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
    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        pass

    @abstractmethod
    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        pass

    def from_grid(self, coordinates: NDArray, axes: Optional[List[str]] = None) -> NDArray:
        """
        Convert Cartesian grid coordinates to the native coordinate system.

        Parameters
        ----------
        coordinates : NDArray
            Array of Cartesian coordinates with shape ``(..., len(axes))``.
        axes : Optional[List[str]], optional
            List of axes corresponding to the coordinates provided. If None, assumes
            the first `len(axes)` axes of the Cartesian coordinate system (e.g., ['x', 'y', 'z'][:len(coordinates.shape[-1])]).

        Returns
        -------
        NDArray
            Array of coordinates in this system's native coordinate system, with shape ``(..., NDIM)``.
        """
        fixed_axes = ['x', 'y', 'z']
        # Determine the axes if not provided
        if axes is None:
            axes = ['x', 'y', 'z'][:coordinates.shape[-1]]

        # Validate input axes
        if any(axis not in fixed_axes for axis in axes):
            raise ValueError(f"Axes {axes} contain invalid axis names. Must be a subset of {fixed_axes}.")

        # Create a full Cartesian coordinate array with missing axes filled with zeros
        full_coordinates = np.zeros((*coordinates.shape[:-1], len(fixed_axes)))
        for i, axis in enumerate(axes):
            full_coordinates[..., fixed_axes.index(axis)] = coordinates[..., i]

        # Convert to the native coordinate system
        return self.from_cartesian(full_coordinates)

    def to_grid(self, coordinates: NDArray, axes: Optional[List[str]] = None) -> NDArray:
        """
        Convert native coordinates in this coordinate system to a Cartesian grid with specified axes.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates in this system's native format, with shape ``(..., NDIM)``.
        axes : Optional[List[str]], optional
            List of axes for the output Cartesian grid. If None, assumes the first `NDIM` axes of the Cartesian coordinate system.

        Returns
        -------
        NDArray
            Array of Cartesian coordinates in the specified axes, with shape ``(..., len(axes))``.
        """
        # Determine the axes if not provided
        if axes is None:
            axes = self.AXES[:self.NDIM]

        # Validate input axes
        fixed_axes = ['x', 'y', 'z']
        if any(axis not in fixed_axes for axis in axes):
            raise ValueError(f"Axes {axes} contain invalid axis names. Must be a subset of {fixed_axes}.")

        # Convert native coordinates to full Cartesian coordinates
        cartesian_coordinates = self.to_cartesian(coordinates)

        # Extract the requested axes from the Cartesian coordinates
        grid_coordinates = np.stack(
            [cartesian_coordinates[..., fixed_axes.index(axis)] for axis in axes],
            axis=-1
        )

        return grid_coordinates

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
            'args': self._args,
            'kwargs': self._kwargs
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
        return _cls(*data['args'], **data['kwargs'])

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
            'args': self._args,
            'kwargs': self._kwargs
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
        return _cls(*data['args'], **data['kwargs'])

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
        group_obj.attrs['args'] = json.dumps(self._args)  # convert args to JSON-compatible format

        # Save each kwarg individually as an attribute
        for key, value in self._kwargs.items():
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
            'args': json.loads(group_obj.attrs['args'])  # deserialize args from JSON-compatible format
        }

        # Load kwargs, deserializing complex data as needed
        kwargs = {}
        for key, value in group_obj.attrs.items():
            if key not in ('class_name', 'args'):
                try:
                    kwargs[key] = json.loads(value)  # try to parse complex JSON data
                except (TypeError, json.JSONDecodeError):
                    kwargs[key] = value  # simple data types remain as is

        data['kwargs'] = kwargs

        _cls = find_in_subclasses(CoordinateSystem, data['class_name'])
        return _cls(*data['args'], **data['kwargs'])

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
            tuple(self._args),
            tuple(sorted(self._kwargs.items()))
        ))