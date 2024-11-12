"""
Symmetry Support Module for Pisces Geometry Handling
====================================================

The cornerstone of symmetry management in the :py:mod:`geometry` module is the :py:class:`Symmetry` class, which
allows the user or developer to specify invariance relative to translations along a coordinate system's axes. This
class provides the logic necessary for manipulating symmetries under various operations, enabling efficient calculations
 and optimizations.

Symmetry
--------

A "symmetry" is characterized very simply: it is effectively a list of axes for which "invariance" holds.
The :py:class:`Symmetry` class formalizes this by taking a list of axes over which the system or object is invariant.

.. hint::

    :py:class:`Symmetry` takes a list of **strings** representing axis names, not axis indices. This is because
    symmetries are **independent of a specific coordinate system** in their implementation. Thus, you specify the
     name of the axes for which invariance holds, and the precise meaning of that will vary between the coordinate
      systems it is applied to. This interplay is managed by the :py:class:`geometry.handlers.GeometryHandler` class.

The :py:class:`Symmetry` class is useful both on its own and in combination with a
:py:class:`geometry.abc.CoordinateSystem` class, where it enables optimized handling of various geometric
 operations. The `Symmetry` class provides a structured framework for understanding and managing invariance,
  including methods to calculate intersections, unions, and other logical operations between different symmetries.

Unified Symmetry and Invariance Management
------------------------------------------

Each instance of the `Symmetry` class inherently knows the invariance state of its associated axes,
enabling easy manipulation and combination of symmetry properties across multiple operations. This unified approach
simplifies the management of symmetrical and asymmetrical dimensions, making it easier to incorporate symmetry into
physical and mathematical models.

Key Operations
--------------

The `Symmetry` class supports a range of operations to facilitate efficient symmetry management:

- **Intersection**: Compute the common axes of invariance between different symmetries.
- **Union**: Combine the axes of invariance from multiple symmetries.
- **Set Difference**: Determine the axes that are invariant in one symmetry but not another.
- **Symmetry Checks**: Quickly assess whether a particular axis is invariant.

These operations provide a powerful means to incorporate symmetry into computational models, optimizing performance and enhancing code readability.
"""
from typing import TYPE_CHECKING, Union, Iterable, Optional, List

import numpy as np

from pisces.geometry._typing import InvarianceArray

if TYPE_CHECKING:
    from pisces.geometry.abc import CoordinateSystem

class Symmetry:
    def __init__(self, symmetry_axes: Iterable[str|int], coordinate_system: 'CoordinateSystem'):
        # Validate inputs and enforce type conventions
        self.symmetry_axes, self.coordinate_system = self._validate_args(symmetry_axes, coordinate_system)

        # Construct additional attributes
        self._construct_attributes()

    @staticmethod
    def _validate_args(symmetry_axes, coordinate_system):
        """
        Validate and convert symmetry axes to integer indices based on the coordinate system.

        Parameters
        ----------
        symmetry_axes : Iterable[str | int]
            Symmetry axes to validate and convert.
        coordinate_system : CoordinateSystem
            The coordinate system providing valid axes.

        Returns
        -------
        set[int], CoordinateSystem
            Validated and converted symmetry axes, and the coordinate system.
        """
        try:
            symmetry_axes = set(
                int(i) if isinstance(i, (int,np.int_)) else coordinate_system.AXES.index(i) for i in symmetry_axes
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid symmetry axes {symmetry_axes}. Must be in {coordinate_system.AXES}."
            ) from e
        return symmetry_axes, coordinate_system

    def _construct_attributes(self):
        """
        Construct a boolean array representing the invariance state for each axis.
        """
        self._invariance_array = np.zeros(self.coordinate_system.NDIM, dtype=bool)
        self._invariance_array[list(self.symmetry_axes)] = True

    @property
    def NDIM(self) -> int:
        """
        The total number of dimensions in the coordinate system.

        This property retrieves the dimensionality of the associated coordinate system, which
        represents the full set of possible axes, regardless of symmetry.

        Returns
        -------
        int
            The total number of dimensions in the coordinate system.

        Examples
        --------

        In a spherical coordinate system with only radial dependence, the ``NDIM`` is 3.

        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym = Symmetry(['phi', 'theta'], cs)
        >>> sym.NDIM
        3

        See Also
        --------
        Symmetry.NDIM_SIM : Returns the number of symmetric (invariant) dimensions.
        Symmetry.NDIM_NSIM : Returns the number of non-symmetric (variant) dimensions.
        Symmetry.is_symmetric: Check the symmetry of an axis.
        """
        return self.coordinate_system.NDIM

    @property
    def NDIM_SIM(self) -> int:
        """
        The number of symmetric (invariant) dimensions.

        This property counts the number of axes in the coordinate system that are part of the
        symmetry, meaning that the object or system is invariant along these axes.

        Returns
        -------
        int
            The number of dimensions along which the symmetry is invariant.

        Examples
        --------

        In a spherical coordinate system with only radial dependence, the ``NDIM_SIM`` is 2.

        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym = Symmetry(['phi', 'theta'], cs)
        >>> sym.NDIM_SIM
        2

        See Also
        --------
        Symmetry.NDIM : Returns the number of dimensions.
        Symmetry.NDIM_NSIM : Returns the number of non-symmetric (variant) dimensions.
        Symmetry.is_symmetric: Check the symmetry of an axis.
        """
        return np.count_nonzero(self._invariance_array)

    @property
    def NDIM_NSIM(self) -> int:
        """
        The number of non-symmetric (variant) dimensions.

        This property calculates the number of axes that are not part of the symmetry,
        indicating the dimensions along which the system varies.

        Returns
        -------
        int
            The number of dimensions along which the symmetry is not invariant.

        Examples
        --------

        In a spherical coordinate system with only radial dependence, the ``NDIM_NSIM`` is 1.

        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym = Symmetry(['phi', 'theta'], cs)
        >>> sym.NDIM_NSIM
        1

        See Also
        --------
        Symmetry.NDIM : Returns the number of dimensions.
        Symmetry.NDIM_SIM : Returns the number of symmetric (invariant) dimensions.
        Symmetry.is_symmetric: Check the symmetry of an axis.
        """
        return self.NDIM - self.NDIM_SIM

    def is_symmetric(self, axis: str|int) -> bool:
        r"""
        Check if the symmetry includes the specified axis.

        This method verifies whether a given axis is part of the symmetry. The axis can be specified
        either by name (string) or by index (integer). If the axis is part of the symmetry, the method
        returns `True`; otherwise, it returns `False`.

        Parameters
        ----------
        axis : str or int
            The name (e.g., 'phi', 'theta') or index (e.g., 0, 1) of the axis to check.

        Returns
        -------
        bool
            `True` if the axis is part of the symmetry, `False` otherwise.

        Examples
        --------
        In a spherical coordinate system with only radial dependence:

        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym = Symmetry(['phi', 'theta'], cs)
        >>> sym.is_symmetric('phi')
        True
        >>> sym.is_symmetric('r')
        False
        >>> sym.is_symmetric(1)  # Assuming 'theta' is the second axis
        True

        Notes
        -----
        - The axis can be provided as a string name or an integer index.
        - This method checks membership in the symmetry set, which defines invariant axes in the
          coordinate system.
        - Axis names must match the names defined in the `CoordinateSystem` associated with the symmetry.
          If an invalid axis name is provided, the method will return `False`.

        See Also
        --------
        Symmetry.NDIM_SIM : Returns the number of symmetric (invariant) dimensions.
        Symmetry.NDIM_NSIM : Returns the number of non-symmetric (variant) dimensions.
        """
        return axis in self

    def __repr__(self) -> str:
        r"""Provide a detailed string representation of the Symmetry instance."""
        return f"<Symmetry: axes={sorted(self.symmetry_axes)}, cs={self.coordinate_system}>"
    def __str__(self) -> str:
        r"""Provide a concise string representation of the Symmetry instance."""
        return f"Symmetry({sorted(self.symmetry_axes)})"
    def __eq__(self, other: Union[InvarianceArray, 'Symmetry']) -> bool:
        other = self._ensure_symmetry(other)
        return (self.symmetry_axes == other.symmetry_axes) and (self.coordinate_system == other.coordinate_system)

    def __len__(self) -> int:
        return self.coordinate_system.NDIM

    def __contains__(self, axis: Union[str, int]) -> bool:
        if isinstance(axis, str):
            try:
                axis: int = int(self.coordinate_system.AXES.index(axis))
            except ValueError:
                return False
        return bool(self._invariance_array[axis])

    def __and__(self, other: Union[InvarianceArray, 'Symmetry']) -> 'Symmetry':
        return self.intersect(other)

    def __or__(self, other: Union[InvarianceArray, 'Symmetry']) -> 'Symmetry':
        return self.union(other)

    def intersect(self, other: Union[InvarianceArray, 'Symmetry']) -> 'Symmetry':
        """
        Compute the intersection of the current symmetry with another.

        This method finds the common axes between the current symmetry and another symmetry
        or invariance array, returning a new `Symmetry` instance representing the intersection.

        Parameters
        ----------
        other : InvarianceArray or Symmetry
            The other symmetry to intersect with. Can be a boolean array representing invariance
            or a `Symmetry` instance.

        Returns
        -------
        Symmetry
            A new `Symmetry` instance representing the intersection of the two symmetries.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi', 'theta'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> sym_intersection = sym1.intersect(sym2)
        >>> sym_intersection.symmetry_axes
        {1}

        Notes
        -----
        - This method raises a `ValueError` if the coordinate systems of the two symmetries are not the same.

        See Also
        --------
        union : Compute the union of two symmetries.
        collection_intersection : Compute the intersection of multiple symmetries.
        """
        other = self._ensure_symmetry(other)
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(
                f"Dissimilar coordinate systems: {self.coordinate_system} vs {other.coordinate_system}."
            )
        return Symmetry(self.symmetry_axes.intersection(other.symmetry_axes), self.coordinate_system)

    def union(self, other: Union[InvarianceArray, 'Symmetry']) -> 'Symmetry':
        """
        Compute the union of the current symmetry with another.

        This method combines the axes of the current symmetry with those of another symmetry
        or invariance array, returning a new `Symmetry` instance representing the union.

        Parameters
        ----------
        other : InvarianceArray or Symmetry
            The other symmetry to unite with. Can be a boolean array representing invariance
            or a `Symmetry` instance.

        Returns
        -------
        Symmetry
            A new `Symmetry` instance representing the union of the two symmetries.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> sym_union = sym1.union(sym2)
        >>> sym_union.symmetry_axes
        {1, 2}

        Notes
        -----
        - This method raises a `ValueError` if the coordinate systems of the two symmetries are not the same.

        See Also
        --------
        intersect : Compute the intersection of two symmetries.
        collection_union : Compute the union of multiple symmetries.
        """
        other = self._ensure_symmetry(other)
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(
                f"Dissimilar coordinate systems: {self.coordinate_system} vs {other.coordinate_system}."
            )
        return Symmetry(self.symmetry_axes.union(other.symmetry_axes), self.coordinate_system)

    def is_superset(self, other: Union[InvarianceArray, 'Symmetry']) -> bool:
        """
        Check if the current symmetry is a superset of another.

        This method verifies if the current symmetry includes all axes of another symmetry
        or invariance array.

        Parameters
        ----------
        other : InvarianceArray or Symmetry
            The other symmetry to compare with. Can be a boolean array representing invariance
            or a `Symmetry` instance.

        Returns
        -------
        bool
            `True` if the current symmetry includes all axes of the other symmetry, `False` otherwise.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi', 'theta'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> sym1.is_superset(sym2)
        False

        Notes
        -----
        - This method raises a `ValueError` if the coordinate systems of the two symmetries are not the same.

        See Also
        --------
        is_subset : Check if the current symmetry is a subset of another.
        """
        other = self._ensure_symmetry(other)
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(
                f"Dissimilar coordinate systems: {self.coordinate_system} vs {other.coordinate_system}."
            )
        return self.symmetry_axes.issubset(other.symmetry_axes)

    def is_subset(self, other: Union[InvarianceArray, 'Symmetry']) -> bool:
        """
        Check if the current symmetry is a subset of another.

        This method verifies if the current symmetry is entirely contained within another symmetry
        or invariance array.

        Parameters
        ----------
        other : InvarianceArray or Symmetry
            The other symmetry to compare with. Can be a boolean array representing invariance
            or a `Symmetry` instance.

        Returns
        -------
        bool
            `True` if the current symmetry is entirely contained within the other symmetry, `False` otherwise.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['theta'], cs)
        >>> sym2 = Symmetry(['phi', 'theta'], cs)
        >>> sym1.is_subset(sym2)
        False

        Notes
        -----
        - This method raises a `ValueError` if the coordinate systems of the two symmetries are not the same.

        See Also
        --------
        is_superset : Check if the current symmetry is a superset of another.
        """
        other = self._ensure_symmetry(other)
        if not self.coordinate_system == other.coordinate_system:
            raise ValueError(
                f"Dissimilar coordinate systems: {self.coordinate_system} vs {other.coordinate_system}."
            )
        return self.symmetry_axes.issuperset(other.symmetry_axes)

    @classmethod
    def collection_intersection(cls, symmetries: Iterable[Union[InvarianceArray, 'Symmetry']],
                                coordinate_system=None) -> 'Symmetry':
        """
        Compute the intersection of multiple symmetries.

        This method takes an iterable of symmetries or invariance arrays and computes the intersection,
        returning a new `Symmetry` instance.

        Parameters
        ----------
        symmetries : Iterable[Union[InvarianceArray, Symmetry]]
            An iterable of symmetries or invariance arrays to intersect.
        coordinate_system : CoordinateSystem, optional
            The coordinate system for the invariance arrays. Required if any element is an `InvarianceArray`.

        Returns
        -------
        Symmetry
            A new `Symmetry` instance representing the intersection of all provided symmetries.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> sym3 = Symmetry(['phi', 'theta'], cs)
        >>> result = Symmetry.collection_intersection([sym1, sym2, sym3])
        >>> result.symmetry_axes
        set()

        Notes
        -----
        - This method raises a `ValueError` if the symmetries have different coordinate systems.

        See Also
        --------
        intersect : Compute the intersection of two symmetries.
        collection_union : Compute the union of multiple symmetries.
        """
        symmetries = [elmt if isinstance(elmt, Symmetry) else cls.from_array(elmt, coordinate_system) for elmt in
                      symmetries]
        if len(set(sym.coordinate_system for sym in symmetries)) != 1:
            raise ValueError("All symmetries must have the same coordinate system for intersection.")

        _initial_symmetry = symmetries.pop(0)
        for _symmetry in symmetries:
            _initial_symmetry = _initial_symmetry.intersect(_symmetry)
        return _initial_symmetry

    @classmethod
    def collection_union(cls, symmetries: Iterable[Union[InvarianceArray, 'Symmetry']],
                         coordinate_system=None) -> 'Symmetry':
        """
        Compute the union of multiple symmetries.

        This method takes an iterable of symmetries or invariance arrays and computes the union,
        returning a new `Symmetry` instance.

        Parameters
        ----------
        symmetries : Iterable[Union[InvarianceArray, Symmetry]]
            An iterable of symmetries or invariance arrays to unite.
        coordinate_system : CoordinateSystem, optional
            The coordinate system for the invariance arrays. Required if any element is an `InvarianceArray`.

        Returns
        -------
        Symmetry
            A new `Symmetry` instance representing the union of all provided symmetries.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> sym3 = Symmetry(['r'], cs)
        >>> result = Symmetry.collection_union([sym1, sym2, sym3])
        >>> result.symmetry_axes
        {0, 1, 2}

        Notes
        -----
        - This method raises a `ValueError` if the symmetries have different coordinate systems.

        See Also
        --------
        union : Compute the union of two symmetries.
        collection_intersection : Compute the intersection of multiple symmetries.
        """
        symmetries = [elmt if isinstance(elmt, Symmetry) else cls.from_array(elmt, coordinate_system) for elmt in
                      symmetries]
        if len(set(sym.coordinate_system for sym in symmetries)) != 1:
            raise ValueError("All symmetries must have the same coordinate system for intersection.")

        _initial_symmetry = symmetries.pop(0)
        for _symmetry in symmetries:
            _initial_symmetry = _initial_symmetry.union(_symmetry)
        return _initial_symmetry

    def set_minus(self, other: Union[InvarianceArray, 'Symmetry']) -> 'Symmetry':
        """
        Compute the set difference (intersection) of the current symmetry with another.

        This method effectively computes the intersection of the current symmetry with another,
        returning a new `Symmetry` instance representing the result.

        Parameters
        ----------
        other : InvarianceArray or Symmetry
            The other symmetry to compare with. Can be a boolean array representing invariance
            or a `Symmetry` instance.

        Returns
        -------
        Symmetry
            A new `Symmetry` instance representing the set difference.

        Examples
        --------
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
        >>> from pisces.geometry.symmetry import Symmetry
        >>> cs = SphericalCoordinateSystem()
        >>> sym1 = Symmetry(['phi', 'theta'], cs)
        >>> sym2 = Symmetry(['theta'], cs)
        >>> result = sym1.set_minus(sym2)
        >>> result.symmetry_axes
        {1}

        Notes
        -----
        - This method raises a `ValueError` if the coordinate systems of the two symmetries are not the same.

        See Also
        --------
        intersect : Compute the intersection of two symmetries.
        """
        return self.intersect(other)

    def partial_derivative(self, axis: int) -> 'Symmetry':
        r"""
        Compute the invariance state after taking the partial derivative along a specified axis.

        This method assesses how the invariance state of a field changes when a partial derivative
        is taken along a particular axis. If the axis corresponds to an invariance axis of the symmetry,
        then a full symmetry (all axes invariant) is returned. Otherwise the symmetry is unaffected.

        Parameters
        ----------
        axis : int
            The index of the axis along which to compute the partial derivative.

        Returns
        -------
        Symmetry
            A new ``Symmetry`` object representing the updated invariance state.

        Notes
        ------------------------
        Let :math:`q^1, \cdots, q^N` be an :math:`N`-dimensional coordinate system and let :math:`\phi` be a
        scalar field on the space spanned by that coordinate system. By definition, an axis :math:`q^k` is in
        the invariance array :math:`Q(\phi)` if and only if :math:`\partial_k \phi = 0` for all points in space.
        Therefore, if

        .. math::

            \phi \in {\rm Inv}_{q^k} \implies \forall {\bf x} \in \mathbb{R}^N,\;\partial_k \phi = 0.

        Thus,

        .. math::

            \phi \in {\rm Inv}_{q^k} \implies \partial_k \phi \in {\rm Inv}^*,

        where :math:`{\rm Inv}^*` is the complete invariance set for the coordinate system.

        In the case that

        .. math::

            \phi \notin {\rm Inv}_{q^k},

        taking the partial derivative doesn't introduce novel dependence. This can be seen (in a special, but common case)
        as a result of Schwarz's Theorem:

        .. math::

            \phi \notin {\rm Inv}_{q^k} \implies \forall q^l \;\text{s.t.}\; \phi \in {\rm Inv}_{q^l}, \partial_l \partial_k \phi = \partial_k \partial_l \phi = 0.

        As such, the symmetry in :math:`l` is preserved.

        Examples
        --------

        In cylindrical coordinates, we might have symmetry in :math:`\rho` and :math:`z`, but not in
        :math:`\phi`. Then the symmetries are as follows:

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho', 'z'], cs)
        >>> sym.partial_derivative(1)
        <Symmetry: axes=[0, 2], cs=CylindricalCoordinateSystem()>

        For an invariant axis:

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho', 'z'], cs)
        >>> sym.partial_derivative(0)
        <Symmetry: axes=[0, 1, 2], cs=CylindricalCoordinateSystem()>

        See Also
        --------
        gradient_component : Computes the gradient and returns the updated invariance state.
        divergence_component : Calculates the divergence, considering the basis effects on invariance.
        """
        if axis in self.symmetry_axes:
            return self._full()
        return self._similar()

    def gradient_component(self, axis: int, basis='unit') -> 'Symmetry':
        r"""
        Compute the invariance state of a gradient component along a specified axis.

        This method evaluates the effect on invariance when calculating a component of the gradient
        in either the unit or contravariant basis.

        Parameters
        ----------
        axis : int
            The index of the axis for which to compute the gradient component.
        basis : {'unit', 'contravariant','covariant'}, optional
            The basis in which to compute the gradient. Default is 'unit'.

            .. hint::

                When in doubt, you're probably interested in using ``'unit'``, which is why it's the
                default. The difference between each is the scaling factor on the vector. The length of
                the vectors in the covariant basis are the Lame coefficients, and the lengths in the contravariant
                basis are 1 over the Lame coefficients.

                In some special calculations, specific bases may prevent symmetry breaking.

        Returns
        -------
        Symmetry
            A new ``Symmetry`` object representing the updated invariance state.

        Notes
        ------------------------
        For a scalar field :math:`\phi` in orthogonal coordinates, the gradient is defined as:

        .. math::

            \nabla \phi = \sum_i \mathbf{e}^i \partial_i \phi,

        where :math:`\mathbf{e}^i` is the contravariant basis vector. In this form, invariance
        along invariant axes is preserved because :math:`\partial_i \phi = 0` along invariant
        directions. In the unit basis, however, we write the gradient as:

        .. math::

            \nabla \phi = \sum_i \frac{\hat{\mathbf{e}}_i}{\lambda_i} \partial_i \phi,

        where each component is scaled by the corresponding Lame coefficient :math:`\lambda_i`.
        Invariant axes remain unchanged unless :math:`\lambda_i` itself varies along those
        directions, potentially reducing invariance.

        The choice of basis significantly impacts the resulting invariance state, particularly
        when Lame coefficients vary across different axes.

        Examples
        --------

        In cylindrical coordinates, the Lame coefficients are :math:`1`, :math:`r`, and :math:`1` again, so the
        gradient operator is

        .. math::

            \nabla \phi = e^r \partial_r \phi + e^\varphi \partial_\varphi \phi + e^z \partial_z \phi

        in the contravariant basis. Thus, if we have :math:`\phi` symmetric in :math:`r`, then that symmetry is preserved
        in this basis, no :math:`r` dependence is introduced.

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho'], cs)
        >>> sym.gradient_component(1, basis='contravariant')
        <Symmetry: axes=[0], cs=CylindricalCoordinateSystem()>

        In the unit basis,

        .. math::

            \nabla \phi = \hat{e}_r \partial_r \phi + \frac{\hat{e}_\varphi}{r} \partial_\varphi \phi + \hat{e}_z \partial_z \phi,

        thus, we have introduced dependence on :math:`r` and broken the symmetry in order to maintain a unit basis.

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho'], cs)
        >>> sym.gradient_component(1, basis='unit')
        <Symmetry: axes=[], cs=CylindricalCoordinateSystem()>

        See Also
        --------
        partial_derivative : Computes the partial derivative along a specified axis.
        divergence_component : Calculates the divergence, considering the basis effects on invariance.
        """
        if self._invariance_array[axis]:
            return self._similar()
        if basis == 'contravariant':
            return self.partial_derivative(axis)
        return self.set_minus(self.coordinate_system.lame_coefficients[axis].invariance)


    def gradient(self,
                 basis='unit',
                 active_axes: Optional[List[int]] = None) -> 'Symmetry':
        r"""
        Compute the invariance state of the full gradient.

        This method calculates the updated invariance state when the full gradient of a scalar
        field is computed in a specified basis.

        Parameters
        ----------
        basis : {'unit', 'contravariant'}, optional
            The basis in which to compute the gradient. Default is 'unit'.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.

        Returns
        -------
        Symmetry
            A new ``Symmetry`` object representing the updated invariance state.

        Notes
        ------------------------
        The gradient operator, when represented in the contravariant basis as:

        .. math::

            \nabla \phi = \sum_i \mathbf{e}^i \partial_i \phi,

        preserves invariance along all invariant directions of :math:`\phi` because
        :math:`\partial_i \phi = 0` for directions in which :math:`\phi` is invariant.

        In the unit basis, the gradient’s components are scaled by :math:`1/\lambda_i`:

        .. math::

            \nabla \phi = \sum_i \frac{\hat{\mathbf{e}}_i}{\lambda_i} \partial_i \phi.

        As a result, the invariance in each direction can be reduced if :math:`\lambda_i`
        varies along that direction, modifying the original invariance properties of the field.


        The basis choice can significantly affect the computation of gradient components,
        particularly in non-Cartesian coordinate systems.

        Examples
        --------

        In cylindrical coordinates, the Lame coefficients are :math:`1`, :math:`r`, and :math:`1` again, so the
        gradient operator is

        .. math::

            \nabla \phi = e^r \partial_r \phi + e^\varphi \partial_\varphi \phi + e^z \partial_z \phi

        in the contravariant basis. Thus, if we have :math:`\phi` symmetric in :math:`r`, then that symmetry is preserved
        in this basis, no :math:`r` dependence is introduced.

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho', 'z'], cs)
        >>> sym.gradient(basis='covariant')
        <Symmetry: axes=[0,2], cs=CylindricalCoordinateSystem()>

        In the unit basis,

        .. math::

            \nabla \phi = \hat{e}_r \partial_r \phi + \frac{\hat{e}_\varphi}{r} \partial_\varphi \phi + \hat{e}_z \partial_z \phi,

        thus, we have introduced dependence on :math:`r` and broken the symmetry in order to maintain a unit basis.

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho','z'], cs)
        >>> sym.gradient(basis='unit')
        <Symmetry: axes=[2], cs=CylindricalCoordinateSystem()>

        See Also
        --------
        gradient_component : Computes the gradient component along a specific axis.
        divergence : Computes the full divergence, considering the effects on invariance.
        """
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        return Symmetry.collection_intersection(
            self.gradient_component(axis, basis=basis) for axis in active_axes
        )

    def divergence_component(self, axis: int,
                             basis='unit') -> 'Symmetry':
        r"""
        Compute the invariance state of a divergence component along a specified axis.

        This method evaluates the change in the invariance state when calculating a component of
        the divergence in either the unit or contravariant basis. The divergence operation combines
        derivatives with Lame coefficients and the Jacobian determinant, impacting the invariance
        state differently depending on the chosen basis.

        Parameters
        ----------
        axis : int
            The index of the axis along which to compute the divergence component.
        basis : {'unit', 'contravariant'}, optional
            The basis in which to compute the divergence. Default is 'unit'.

            .. hint::

                The 'unit' basis scales the divergence by the inverse of the Lame coefficients,
                potentially affecting invariance along axes where these coefficients vary.
                The 'contravariant' basis avoids this scaling, preserving invariance more effectively.

        Returns
        -------
        Symmetry
            A new ``Symmetry`` object representing the updated invariance state after computing
            the divergence component along the specified axis.

        Notes
        ------------------------
        In orthogonal coordinates, the divergence of a vector field :math:`\mathbf{A} = A^i \mathbf{e}_i`
        is given by:

        .. math::

            \nabla \cdot \mathbf{A} = \frac{1}{J} \sum_i \frac{\partial}{\partial q^i}
            \left( \hat{A}_i \, J \, \frac{1}{\lambda_i} \right),

        where :math:`J = \prod_{i} \lambda_i` is the Jacobian determinant. The invariance state
        depends on the interplay between the partial derivatives and the Lame coefficients. In
        the contravariant basis, where scaling is minimal, the original invariance properties are
        more likely to be preserved.

        Examples
        --------
        In cylindrical coordinates, the Lame coefficients are :math:`1`, :math:`r`, and :math:`1` again. The divergence
        operator becomes:

        .. math::

            \nabla \cdot \mathbf{A} = \frac{1}{r} \frac{\partial}{\partial r} (r A^r) + \frac{1}{r} \frac{\partial A^\varphi}{\partial \varphi} + \frac{\partial A^z}{\partial z}.

        Thus, if a vector field has symmetry in :math:`r` and :math:`z`, it maintains the :math:`z` symmetry, but
        the :math:`r` symmetry may be lost due to the presence of :math:`1/r` scaling in the divergence computation.

        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho', 'z'], cs)
        >>> sym.divergence_component(1)
        <Symmetry: axes=[2], cs=CylindricalCoordinateSystem()>

        See Also
        --------
        divergence : Computes the full divergence, considering the basis effects on invariance.
        gradient_component : Evaluates the invariance state of a gradient component along a specific axis.
        """
        # Copy the invariance state and correct it for the conversion to contravariant
        # basis (removal of the lame coefficient invariance set).
        _invariance_state = self._invariance_array[...]
        if basis != 'contravariant':  # Adjust for non-contravariant basis
            _invariance_state = _invariance_state & self.coordinate_system.lame_coefficients[axis].invariance

        # For this axis, we need to determine the dependence of the Lame coefficients that
        # make up the jacobian. If they don't depend on axis, they are eliminated.
        # The first step is to determine what jacobian terms are actually present.
        _lame_mask = self.coordinate_system.lame_invariance_matrix[:, axis] # T if invar. over axis, F otherwise.
        _jacobian_invariance_states = [
            self.coordinate_system.lame_coefficients[i].invariance for i in np.arange(self.NDIM)[~_lame_mask]
        ]

        # Now that we know what terms are present, we can determine the dependence.
        _operand_invariance = Symmetry.collection_intersection([_invariance_state]+_jacobian_invariance_states,coordinate_system=self.coordinate_system)
        return _operand_invariance

    def divergence(self, basis='unit',active_axes: Optional[List[int]] = None) -> 'Symmetry':
        r"""
        Compute the invariance state of the full divergence.

        The divergence operation involves summing changes across all axes in the field. Using the
        contravariant basis preserves invariance in unaffected directions by minimizing reliance on
        Lame coefficients. In the unit basis, however, the need to scale by Lame coefficients may
        reduce invariance in directions where these coefficients vary.

        Parameters
        ----------
        basis : {'unit', 'contravariant'}, optional
            Basis used to compute divergence. Default is 'unit'.
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.

        Returns
        -------
        InvarianceArray
            A new ``InvarianceArray`` representing the invariance state of the full divergence.

        Examples
        --------

        >>> # Import the Cylindrical coordinate system from pisces.
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho','z'],cs)
        >>> sym.divergence_component(1)
        <Symmetry: axes=[2], cs=CylindricalCoordinateSystem()>

        Mathematical Explanation
        ------------------------
        In orthogonal coordinates, the divergence of a vector field :math:`\mathbf{A} = A^i \mathbf{e}_i`
        is expressed as:

        .. math::

            \nabla \cdot \mathbf{A} = \frac{1}{J} \sum_i \frac{\partial}{\partial q^i}
            \left( A^i \, J \, \frac{1}{\lambda_i} \right),

        where :math:`J = \prod_{i} \lambda_i` represents the Jacobian determinant. When using the
        contravariant basis, the invariance properties are preserved, minimizing dependencies on
        Lame coefficients. In the unit basis, scaling by :math:`1/\lambda_i` along each axis reduces
        invariance where the Lame coefficients vary, showing the geometric dependencies.
        """
        if active_axes is None:
            active_axes = np.arange(self.NDIM)

        return Symmetry.collection_intersection(
            self.divergence_component(axis, basis=basis) for axis in active_axes
        )
    def laplacian(self,active_axes: Optional[List[int]] = None) -> 'Symmetry':
        """
        Compute the invariance state of the Laplacian by calculating the divergence of the gradient
        in the contravariant basis.

        The Laplacian represents the sum of second-order derivatives across all axes, capturing how
        a scalar field’s gradient flux varies within a volume. This method calculates the Laplacian
        by first computing the gradient (in the contravariant basis) and then taking its divergence.

        Parameters
        ----------
        active_axes : Optional[List[int]], optional
            List of axis indices to include in the divergence calculation. By default, all axes are included.

        Returns
        -------
        Symmetry
            A new ``Symmetry`` representing the invariance state of the Laplacian.

        Examples
        --------

        In Cylindrical coordinates, the Laplacian of a field is

        .. math::

            \nabla^2\phi = \frac{1}{\rho}\partial_\rho (\rho \partial_\rho \phi) + \frac{1}{\rho^2} \partial^2_{\varphi} \phi + \partial^2_z \phi.

        Thus, we expect that a field with :math:`\varphi` dependence only will lose its :math:`\rho` invariance after
        the Laplacian operation.

        >>> # Import the Cylindrical coordinate system from pisces.
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry(['rho','z'],cs)
        >>> sym.divergence()
        <Symmetry: axes=[2], cs=CylindricalCoordinateSystem()>

        Mathematical Explanation
        ------------------------
        The Laplacian of a scalar field :math:`\phi` in orthogonal coordinates is defined as:

        .. math::

            \nabla^2 \phi = \frac{1}{J} \sum_i \frac{\partial}{\partial q^i}
            \left( \frac{J}{\lambda_i^2} \frac{\partial \phi}{\partial q^i} \right),

        where :math:`J = \prod_i \lambda_i` is the Jacobian determinant and :math:`\lambda_i` are
        the Lame coefficients. The Laplacian is the divergence of the gradient, and by calculating
        it in the contravariant basis, the calculation avoids scaling that might reduce invariance.

        Invariance is determined by analyzing the dependencies induced by :math:`\lambda_i^2`, where
        the Laplacian maintains maximal invariance by considering the dependencies within each
        gradient and divergence operation.
        """
        return self.gradient(basis='contravariant',active_axes=active_axes).divergence(basis='contravariant',active_axes=active_axes)

    @classmethod
    def from_array(cls, array: InvarianceArray, coordinate_system: 'CoordinateSystem') -> 'Symmetry':
        """
        Create a Symmetry object from an InvarianceArray.

        This class method constructs a `Symmetry` object by converting an `InvarianceArray` to a list
        of invariant axes.

        Parameters
        ----------
        array : InvarianceArray
            An array indicating which axes are invariant.
        coordinate_system : CoordinateSystem
            The coordinate system associated with the invariance.

        Returns
        -------
        Symmetry
            A new `Symmetry` object with invariance properties derived from the input array.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> arr = InvarianceArray([True, False, True])
        >>> sym = Symmetry.from_array(arr, cs)
        >>> print(sym)
        <Symmetry: axes=[0, 2], cs=CylindricalCoordinateSystem()>
        """
        return Symmetry(np.arange(len(array))[array], coordinate_system)

    @classmethod
    def full_symmetry(cls, coordinate_system: 'CoordinateSystem') -> 'Symmetry':
        """
        Create a Symmetry object with full symmetry (all axes invariant).

        This method generates a `Symmetry` object where all axes of the coordinate system
        are considered invariant.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system for which full symmetry is defined.

        Returns
        -------
        Symmetry
            A `Symmetry` object with all axes invariant.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> full_sym = Symmetry.full_symmetry(cs)
        >>> print(full_sym)
        <Symmetry: axes=[0, 1, 2], cs=CylindricalCoordinateSystem()>
        """
        return Symmetry(np.arange(coordinate_system.NDIM), coordinate_system)

    @classmethod
    def empty_symmetry(cls, coordinate_system: 'CoordinateSystem') -> 'Symmetry':
        """
        Create a Symmetry object with no symmetry (no axes invariant).

        This method generates a `Symmetry` object where no axes of the coordinate system
        are considered invariant.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system for which no symmetry is defined.

        Returns
        -------
        Symmetry
            A `Symmetry` object with no axes invariant.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> empty_sym = Symmetry.empty_symmetry(cs)
        >>> print(empty_sym)
        <Symmetry: axes=[], cs=CylindricalCoordinateSystem()>
        """
        return Symmetry([], coordinate_system)

    def _ensure_symmetry(self, other: Union['Symmetry', InvarianceArray]) -> 'Symmetry':
        """
        Ensure an input is a Symmetry object.

        This method converts an `InvarianceArray` to a `Symmetry` object if necessary.
        If the input is already a `Symmetry` object, it is returned as is.

        Parameters
        ----------
        other : Union[Symmetry, InvarianceArray]
            The object to ensure is a `Symmetry` instance.

        Returns
        -------
        Symmetry
            A `Symmetry` object corresponding to the input invariance.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry, InvarianceArray
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> array = InvarianceArray([True, False, True])
        >>> sym = Symmetry([], cs)
        >>> ensured_sym = sym._ensure_symmetry(array)
        >>> print(ensured_sym)
        <Symmetry: axes=[0, 2], cs=CylindricalCoordinateSystem()>
        """
        if isinstance(other, Symmetry):
            return other
        else:
            return self.from_array(other, self.coordinate_system)

    def _full(self) -> 'Symmetry':
        """
        Internal method to return a Symmetry object with full symmetry.

        This method returns a `Symmetry` object where all axes of the coordinate system
        are invariant, intended for internal use.

        Returns
        -------
        Symmetry
            A `Symmetry` object with full symmetry.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry([], cs)
        >>> full_sym = sym._full()
        >>> print(full_sym)
        <Symmetry: axes=[0, 1, 2], cs=CylindricalCoordinateSystem()>
        """
        return self.full_symmetry(self.coordinate_system)

    def _empty(self) -> 'Symmetry':
        """
        Internal method to return a Symmetry object with no symmetry.

        This method returns a `Symmetry` object with no axes invariant, intended for internal use.

        Returns
        -------
        Symmetry
            A `Symmetry` object with no symmetry.

        Examples
        --------
        >>> from pisces.geometry.symmetry import Symmetry
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
        >>> cs = CylindricalCoordinateSystem()
        >>> sym = Symmetry([], cs)
        >>> empty_sym = sym._empty()
        >>> print(empty_sym)
        <Symmetry: axes=[], cs=CylindricalCoordinateSystem()>
        """
        return self.empty_symmetry(self.coordinate_system)

    def _similar(self) -> 'Symmetry':
        """ Create a copy of symmetry of this instance."""
        return Symmetry(list(self.symmetry_axes),self.coordinate_system)

