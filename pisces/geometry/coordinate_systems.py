r"""
Coordinate Systems for Pisces

This module contains the coordinate systems for the Pisces library. This includes various spheroidal
coordinate systems, cartesian coordinate systems, cylindrical coordinate systems, and others. For each of
these coordinate systems, structures have been generated to allow for differential operations and the tracking
of symmetries through those operations.

For users unfamiliar with this module, we suggest reading both :ref:`geometry_overview` and :ref:`geometry_theory`.
"""

from typing import TYPE_CHECKING, Callable, Union

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.integrate import dblquad, quad
from scipy.interpolate import InterpolatedUnivariateSpline

from pisces.geometry.base import CoordinateSystem, RadialCoordinateSystem
from pisces.utilities.logging import mylog
from pisces.utilities.math_utils.numeric import integrate

if TYPE_CHECKING:
    pass


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class CartesianCoordinateSystem(CoordinateSystem):
    r"""
    3 Dimensional Cartesian coordinate system.

    The Cartesian coordinate system represents a flat, orthogonal system without curvature. It is defined by the coordinates
    :math:`(x, y, z)` in 3D space. In this system, each coordinate axis is a straight line, and all basis vectors are unit vectors,
    meaning that the system does not scale or curve.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x, y, z) \to (x, y, z) \\
           \text{From Cartesian:} & \quad (x, y, z) \to (x, y, z)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    | :math:`y`| :math:`1`               |
    +----------+-------------------------+
    | :math:`z`| :math:`1`               |
    +----------+-------------------------+

    Examples
    --------

    We can now plot the function:

    .. plot::
        :include-source:

        Let's begin by initializing the Cartesian coordinate system:

        >>> from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
        >>> coordinate_system = CartesianCoordinateSystem()

        We can now initialize a Cartesian grid. We'll use a slice in X-Y:

        >>> grid = np.mgrid[-1:1:100j,-1:1:100j,-1:1:3j]
        >>> grid = np.moveaxis(grid,0,-1) # fix the grid ordering to meet our standard

        Let's now create a function on this geometry.

        >>> func = lambda x,y: np.cos(y)*np.sin(x*y)
        >>> Z = func(grid[...,0],grid[...,1])

        >>> import matplotlib.pyplot as plt
        >>> image_array = Z[:,:,1].T
        >>> plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP

    Let's now compute the gradient of this scalar field! This will produce all 3 components of the
    gradient. We'll visualize the :math:`x` direction.

    .. plot::
        :include-source:

        >>> from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
        >>> coordinate_system = CartesianCoordinateSystem()
        >>> grid = np.mgrid[-1:1:100j,-1:1:100j,-1:1:3j]
        >>> grid = np.moveaxis(grid,0,-1) # fix the grid ordering to meet our standard
        >>> func = lambda x,y: np.cos(y)*np.sin(x*y)
        >>> Z = func(grid[...,0],grid[...,1])
        >>> gradZ = coordinate_system.compute_gradient(Z,grid)
        >>> image_array = gradZ[:,:,1,0].T
        >>> plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP
        >>> plt.show()

    """
    NDIM = 3
    AXES = ["x", "y", "z"]
    PARAMETERS = {}

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Cartesian is already in native form

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Cartesian is already in native form

    def lame_0(x, y, z):
        return 1

    def lame_1(x, y, z):
        return 1

    def lame_2(x, y, z):
        return 1


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class CartesianCoordinateSystem1D(CoordinateSystem):
    r"""
    1 Dimensional Cartesian coordinate system.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x) \to (x) \\
           \text{From Cartesian:} & \quad (x) \to (x)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    """
    NDIM = 1
    AXES = ["x"]
    PARAMETERS = {}

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def lame_0(x):
        return 1


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class CartesianCoordinateSystem2D(CoordinateSystem):
    r"""
    2 Dimensional Cartesian coordinate system.

    Conversion to and from Cartesian coordinates is trivial as the coordinates are already in this form:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad (x,y) \to (x,y) \\
           \text{From Cartesian:} & \quad (x,y) \to (x,y)
       \end{aligned}

    Notes
    -----
    Lame coefficients for each axis in the Cartesian coordinate system are all equal to 1. This lack of scaling factors is
    why the system is often preferred in Euclidean geometry calculations.

    +----------+-------------------------+
    | Axis     | Lame Coefficient        |
    +==========+=========================+
    | :math:`x`| :math:`1`               |
    +----------+-------------------------+
    | :math:`y`| :math:`1`               |
    +----------+-------------------------+

    Examples
    --------
    The Cartesian 2D coordinate system is visualized as a regular grid of constant x and y values.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import CartesianCoordinateSystem2D

        Initialize the coordinate system:

        >>> coordinate_system = CartesianCoordinateSystem2D()

        Define a grid of points:

        >>> x_vals = np.linspace(-1, 1, 10)  # x values
        >>> y_vals = np.linspace(-1, 1, 10)  # y values
        >>> x, y = np.meshgrid(x_vals, y_vals)

        Plot the grid:

        >>> for i in range(len(x_vals)):
        ...     _ = plt.plot(x[:, i], y[:, i], 'k-', lw=0.5)
        ...     _ = plt.plot(x[i, :], y[i, :], 'k-', lw=0.5)

        >>> _ = plt.title('Cartesian 2D Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    NDIM = 2
    AXES = ["x", "y"]
    PARAMETERS = {}

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def lame_0(x, y):
        return 1

    def lame_1(x, y):
        return 1


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class SphericalCoordinateSystem(RadialCoordinateSystem):
    r"""
    3 Dimensional Spherical coordinate system.

    The spherical coordinate system is defined by the coordinates :math:`(r, \theta, \phi)`, where :math:`r` is the radial distance,
    :math:`\theta` is the polar angle, and :math:`\phi` is the azimuthal angle. This system is ideal for spherical symmetries,
    such as in gravitational and electrostatic fields.

    Conversion between spherical and Cartesian coordinates follows these standard transformations:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = r \sin(\theta) \cos(\phi), \\
                                & \quad y = r \sin(\theta) \sin(\phi), \\
                                & \quad z = r \cos(\theta) \\
           \text{From Cartesian:} & \quad r = \sqrt{x^2 + y^2 + z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{r}\right), \\
                                  & \quad \phi = \arctan2(y, x)
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the spherical coordinate system are provided in the table below, reflecting the
    scaling factors due to curvature in radial and angular directions.

    +-------------------+------------------------------+
    | Axis              | Lame Coefficient             |
    +===================+==============================+
    |:math:`r`          | :math:`1`                    |
    +-------------------+------------------------------+
    | :math:`\theta`    | :math:`r`                    |
    +-------------------+------------------------------+
    | :math:`\phi`      | :math:`r \sin(\theta)`       |
    +-------------------+------------------------------+

    Examples
    --------
    The Spherical coordinate system is visualized with circles (constant r) and lines radiating from the origin (constant theta) on a 2D slice.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = SphericalCoordinateSystem()

        Define radial and angular ranges:

        >>> r_vals = np.linspace(0, 1, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, np.pi, 12)  # Angular values
        >>> phi = 0  # Fix the azimuthal angle

        Plot circles (constant r):

        >>> for r in r_vals:
        ...     theta = np.linspace(0, np.pi, 200)
        ...     coords = np.stack([r * np.ones_like(theta), theta, np.full_like(theta, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        Plot radial lines (constant theta):

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 1, 200)
        ...     coords = np.stack([r, theta * np.ones_like(r), np.full_like(r, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        >>> _ = plt.title('Spherical Coordinate System (Slice)')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    NDIM = 3
    AXES = ["r", "theta", "phi"]
    PARAMETERS = {}

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

    def lame_0(r, theta, phi):
        return 1

    def lame_1(r, theta, phi):
        return r

    def lame_2(r, theta, phi):
        return r * sp.sin(theta)

    def integrate_in_shells(
        self, field: Union[np.ndarray, Callable], radii: np.ndarray
    ):
        r"""

        Parameters
        ----------
        field
        radii

        Returns
        -------

        Examples
        --------

        Let's calculate the mass of a constant density sphere of density 1. To start, let's set up
        the coordinate system and the arrays.

        .. plot::
            :include-source:

            >>> import numpy as np
            >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
            >>> cs = SphericalCoordinateSystem()
            >>> r = np.linspace(0,1,1000)
            >>> density = np.ones_like(r)

            Now we can do the computation

            >>> mass = cs.integrate_in_shells(density, r)

            Now, the mass should go as

            .. math::

                M(<r) = 4\pi \int_0^r \rho r^2 dr = \frac{4}{3}\pi r^3 = \frac{4}{3}\pi r^3

            >>> import matplotlib.pyplot as plt
            >>> _ = plt.plot(r,mass)
            >>> _ = plt.plot(r, (4/3)*np.pi*r**3)
            >>> plt.show()


        """
        # Construct the integrand and the spline
        if callable(field):
            f = lambda r: field(r) * r**2
        else:
            f = InterpolatedUnivariateSpline(radii, field * radii**2)
        return 4 * np.pi * integrate(f, radii, radii[0], minima=True)

    def solve_radial_poisson_problem(
        self,
        density_profile: Union[np.ndarray, Callable],
        radii: NDArray,
        boundary_mode: str = "integrate",
        powerlaw_index: int = None,
    ):
        r"""
        Solve Poisson's equation (in Planck units) for spherical coordinates, given a density profile.

        This method computes the gravitational potential :math:`\Phi(r)` for a given density profile
        :math:`\rho(r)` in spherical coordinates by solving Poisson's equation:

        .. math::
            \nabla^2 \Phi = 4 \pi \rho(r).

        The solution is returned in Planck units, and therefore has natural units of :math:`[M]/[L]`.
        The density profile should be supplied in units of :math:`[M]/[L]^3` and the radii in
        units of :math:`[L]`. This ensures compatibility with the computational framework.

        Parameters
        ----------
        density_profile : Union[Callable, np.ndarray]
            The density profile of the system, which can either be:

            - A callable function that takes a single argument (``radius``) and returns the density.
            - A numerical array of density values corresponding to the ``radii``.

        radii : np.ndarray
            Array of radii at which the potential is computed. Must be in ascending order.
        boundary_mode : str, optional
            Method for handling the outer boundary. Available options are:

            - ``'integrate'``: The boundary is treated by numerical integration to infinity. This is a viable option
              when the full functional form of ``density_profile`` is provided. If ``density_profile`` is a spline or
              some other interpolation product, then ``'integrate'`` will lead to inaccurate boundary behavior.

            - ``'asymptotic'``: An asymptotic approximation is used beyond the provided radii. Formally, we assume that
              beyond ``radii[-1]``, the density profile goes as :math:`\rho(r)\sim r^l,\; (l < -2)`. In this case,

              .. math::

                  \rho(r) = \rho(r_0) \left(\frac{r}{r_0}\right)^l,

              and

              .. math::

                  \int_{r^0}^{\infty} r \rho(r) \; dr = \frac{\rho(r_0)}{r_0^l} \int_{r^0}^{\infty} r^{l+1}\; dr = - \frac{\rho(r_0)r_0^2}{l+2}.

            Default is `'integrate'`.
        powerlaw_index : int, optional
            Specifies the asymptotic power-law behavior of the density profile at large radii.
            Required only if ``boundary_mode`` is ``'asymptotic'``. This must be a value smaller than :math:`-2`.

        Returns
        -------
        potential : np.ndarray
            Array of computed gravitational potential values at each radius in `radii`, in
            units of mass/length.

        Notes
        -----
        - The computed gravitational potential accounts for contributions from mass enclosed
          at smaller radii (inner integral) and contributions from outer shells (outer integral).
        - When using ``'asymptotic'`` for the ``boundary_mode``, the provided ``boundary_behavior``
          must accurately describe the power-law decline of the density profile at large radii.
          Incorrect values may lead to unphysical results.
        - The density profile :math:`\rho(r)` must be smooth and continuous for accurate results.

        Theory
        ------
        This implementation is based on the solution to Poisson's equation for spherical systems:

        .. math::
            \Phi(r) = -4 \pi G \left[
                \frac{1}{r} \int_0^r r'^2 \rho(r') dr' +
                \int_r^\infty r' \rho(r') dr'
            \right].

        - The first term accounts for the mass enclosed within radius :math:`r`.
        - The second term represents the contribution from mass at larger radii.
        - For more details, refer to the section on Poisson's equation in the theory documentation:
          :ref:`poisson_equation`.

        Examples
        --------
        Compute the potential for a density profile with exponential decay:

        See Also
        --------
        - :py:func:`integrate_from_zero`: Computes definite integrals from 0 to a specified upper bound.
        - :py:func:`cumulative_trapezoid`: Implements trapezoidal rule for cumulative integration.
        - :ref:`poisson_equation`: Detailed derivations and examples of Poisson's equation solutions.

        Warnings
        --------
        - Numerical instabilities may occur if the density profile is poorly sampled or discontinuous.
        - Ensure the density profile units are consistent with the Planck units used in the calculations.

        Examples
        --------
        As an example of this method, let's compute the potential of the Hernquist density profile:

        .. math::

            \rho(r) = \frac{\rho_0}{\xi(1+\xi)^3},

        where :math:`\xi = r/r_s`.

        This has a known solution:

        .. math::

            \Phi(r) = -2\pi\frac{G\rho_0 r_s^3}{r+r_s}.

        In this case, we'll use 1 for :math:`\rho_0` and :math:`r_s`, so

        .. math::

            \rho(r) = \frac{1}{r(1+r)^3},\implies \Phi(r) = -2\pi \frac{1}{1+r}.

        .. plot::
            :include-source:

            >>> from pisces.profiles.density import HernquistDensityProfile
            >>> import numpy as np
            >>> dp = HernquistDensityProfile(rho_0=1,r_s=1)
            >>> r = np.geomspace(1e-1,1e3,10000)
            >>> known_potential = -2*np.pi/(1+r)

            Let's now initialize the geometry and compute the solution.

            >>> from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
            >>> cs = SphericalCoordinateSystem()
            >>> computed_potential = cs.solve_radial_poisson_problem(dp, r)

            We can now make a plot:

            >>> import matplotlib.pyplot as plt
            >>> fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw=dict(hspace=0,height_ratios=[1,0.25]))
            >>> _ = axes[0].loglog(r,-known_potential,label='True')
            >>> _ = axes[0].loglog(r,-computed_potential, label='Computed')
            >>> _ = axes[1].loglog(r, np.abs((known_potential-computed_potential)/known_potential))
            >>> _ = axes[0].set_ylabel(r"Potential $\left(-\Phi(r)\right)$")
            >>> _ = axes[1].set_ylabel(r"Rel. Err.")
            >>> _ = axes[1].set_xlabel(r"Radius")
            >>> _ = axes[0].legend()
            >>> plt.show()

        """
        from pisces.utilities.math_utils.poisson import solve_poisson_spherical

        return solve_poisson_spherical(
            density_profile,
            radii,
            powerlaw_index=powerlaw_index,
            boundary_mode=boundary_mode,
        )


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class PolarCoordinateSystem(CoordinateSystem):
    r"""
    2 Dimensional Polar coordinate system.

    The polar coordinate system is used for 2D spaces with rotational symmetry, defined by the coordinates :math:`(r, \theta)`,
    where :math:`r` represents the radial distance from the origin and :math:`\theta` represents the angle from a reference axis.

    Conversion between polar and Cartesian coordinates is as follows:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = r \cos(\theta), \\
                                & \quad y = r \sin(\theta) \\
           \text{From Cartesian:} & \quad r = \sqrt{x^2 + y^2}, \\
                                  & \quad \theta = \arctan2(y, x)
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the polar coordinate system provide a scaling factor as shown below.

    +-----------------+-------------------------+
    | Axis            | Lame Coefficient        |
    +=================+=========================+
    | :math:`r`       | :math:`1`               |
    +-----------------+-------------------------+
    | :math:`\theta`  | :math:`r`               |
    +-----------------+-------------------------+

    Examples
    --------
    The Polar coordinate system is visualized as concentric circles (constant r) and radial lines (constant theta).

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import PolarCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = PolarCoordinateSystem()

        Define the radial and angular ranges:

        >>> r_vals = np.linspace(0, 1, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values

        Plot concentric circles (constant r):

        >>> for r in r_vals:
        ...     theta = np.linspace(0, 2 * np.pi, 200)
        ...     coords = np.stack([r * np.ones_like(theta), theta], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)

        Plot radial lines (constant theta):

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 1, 200)
        ...     coords = np.stack([r, theta * np.ones_like(r)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)

        >>> _ = plt.title('Polar Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
        """
    NDIM = 2
    AXES = ["r", "theta"]
    PARAMETERS = {}

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        r, theta = coordinates[..., 0], coordinates[..., 1]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack((x, y), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y = coordinates[..., 0], coordinates[..., 1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.stack((r, theta), axis=-1)

    def lame_0(r, theta):
        return 1

    def lame_1(r, theta):
        return r


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class CylindricalCoordinateSystem(CoordinateSystem):
    r"""
    3 Dimensional Cylindrical coordinate system.

    The cylindrical coordinate system is defined by the coordinates :math:`(\rho, \phi, z)`, where :math:`\rho` is the radial
    distance in the xy-plane, :math:`\phi` is the azimuthal angle, and :math:`z` is the height along the z-axis.
    This system is commonly used in problems with axial symmetry, such as electromagnetic fields around a wire.

    Conversion between cylindrical and Cartesian coordinates is defined as:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \rho \cos(\phi), \\
                                & \quad y = \rho \sin(\phi), \\
                                & \quad z = z \\
           \text{From Cartesian:} & \quad \rho = \sqrt{x^2 + y^2}, \\
                                  & \quad \phi = \arctan2(y, x), \\
                                  & \quad z = z
       \end{aligned}

    Notes
    -----
    The Lame coefficients for each axis in the cylindrical coordinate system are detailed below, reflecting scaling
    factors associated with the radial and angular coordinates.

    +-----------------+----------------------------+
    | Axis            | Lame Coefficient           |
    +=================+============================+
    | :math:`\rho`    | :math:`1`                  |
    +-----------------+----------------------------+
    | :math:`\phi`    | :math:`\rho`               |
    +-----------------+----------------------------+
    | :math:`z`       |  :math:`1`                 |
    +-----------------+----------------------------+

    Examples
    --------
    The Cylindrical coordinate system is visualized as concentric circles (constant rho) and vertical lines (constant z) on a 2D slice.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem

        Initialize the coordinate system:

        >>> coordinate_system = CylindricalCoordinateSystem()

        Define the radial and angular ranges:

        >>> rho_vals = np.linspace(0, 1, 6)  # Radial distances
        >>> phi_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values
        >>> z = 0  # Fix the z-coordinate

        Plot concentric circles (constant rho):

        >>> for rho in rho_vals:
        ...     phi = np.linspace(0, 2 * np.pi, 200)
        ...     coords = np.stack([rho * np.ones_like(phi), phi, np.full_like(phi, z)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)

        Plot radial lines (constant phi):

        >>> for phi in phi_vals:
        ...     rho = np.linspace(0, 1, 200)
        ...     coords = np.stack([rho, phi * np.ones_like(rho), np.full_like(rho, z)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)

        >>> _ = plt.title('Cylindrical Coordinate System (Slice)')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('y')
        >>> _ = plt.axis('equal')
        >>> plt.show()
        """
    NDIM = 3
    AXES = ["rho", "phi", "z"]
    PARAMETERS = {}

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        rho, phi, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return np.stack((rho, phi, z), axis=-1)

    def lame_0(rho, phi, z):
        return 1

    def lame_1(rho, phi, z):
        return rho

    def lame_2(rho, phi, z):
        return 1


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class PseudoSphericalCoordinateSystem(RadialCoordinateSystem):
    r"""
    A generalized coordinate system that replaces the standard Euclidean norm with
    a scaled norm of the form:

    .. math::

        r^2 = \eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2.

    The angular coordinates remain the same as in spherical coordinates; however, the
    :math:`r` iso-surfaces are now ellipsoids instead of spheres.

    **Conversion to and from Cartesian coordinates**:

    Because the angular coordinates are the same as in spherical coordinates, the same basic transformations may
    be used; however, an additional scaling is necessary because :math:`r` is no longer the physical distance. Given
    :math:`(r,\theta,\phi)`, the unit vector in the correct direction has the form

    .. math::

        \hat{r} = \begin{pmatrix}\sin(\theta)\cos(\phi)\\\sin(\theta)\sin(\phi)\\\cos(\theta)\end{pmatrix}

    Now, the effective radius of this vector is

    .. math::

        r(\hat{r})^2 = \eta_x^2 \sin^2\theta \cos^2\phi + \eta_y^2 \sin^2\theta \sin^2\phi + \eta_z^2 \cos^2\theta = \Omega^2(\theta,\phi).

    The relevant scaling is then :math:`r/r(\hat{r})`, so the conversions are as follows:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \frac{r}{\Omega(\theta,\phi)} \sin(\theta) \cos(\phi), \\
                                & \quad y = \frac{r}{\Omega(\theta,\phi)} \sin(\theta) \sin(\phi), \\
                                & \quad z = \frac{r}{\Omega(\theta,\phi)} \cos(\theta), \\
           \text{From Cartesian:} & \quad r = \sqrt{\eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{d}\right), \\
                                  & \quad \phi = \arctan2(y, x),
       \end{aligned}

    where

    .. math::

        d^2 = x^2 + y^2 + z^2.

    **Mathematical Background**:

    For a point :math:`(r, \phi, \theta)` in the pseudo-spherical coordinate system, its Cartesian distance is:

    .. math::

        d^2 = x^2 + y^2 + z^2 = \frac{r^2}{\Omega(\theta,\phi)^2}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = \frac{r}{\Omega(\phi, \theta)} \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \frac{1}{\Omega(\phi, \theta)}.

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative. Because
       :math:`\partial_\theta \hat{r} = \hat{\theta}`, we may express this as

       .. math::

            \lambda_\theta = \left| \frac{\partial {\bf r}}{\partial \theta} \right| = \frac{r}{\Omega}\left[1 + \frac{1}{\Omega^2}\left(\frac{\partial \Omega}{\partial \theta}\right)^2\right]^{1/2}

       where

       .. math::

            \frac{\partial \Omega}{\partial \theta} = \frac{1}{\Omega}\sin \theta \cos \theta \left[\eta_x^2\cos^2\phi + \eta_y^2\sin^2\phi - \eta_z^2\right].

       .. note::

           In the special case :math:`\eta_x=\eta_y=\eta_z`, :math:`\Omega = 1` and :math:`\partial_\theta \Omega = 0`, so
           the entire right hand term under the radical is zero and we get

           .. math::
               \left|\frac{\partial {\bf r}}{\partial \theta} \right| = r.


    3. :math:`\phi` **Component**:

       Because :math:`\partial_\phi \hat{r} = \sin\theta \hat{\phi}`, we may express this as

       .. math::

            \lambda_\phi = \left| \frac{\partial {\bf r}}{\partial \phi} \right| = \frac{r}{\Omega}\left[\sin^2\theta + \frac{1}{\Omega^2}\left(\frac{\partial \Omega}{\partial \phi}\right)^2\right]^{1/2}

       where

       .. math::

            \frac{\partial \Omega}{\partial \phi} = \frac{1}{\Omega}\sin^2 \theta \cos \phi \sin \phi \left[\eta_y^2 - \eta_x^2\right].


    **Shell Coefficient**

    A common task in radial coordinates is to integrate over a volume via shells of constant :math:`r`. To do so, one needs
    to know the infinitesimal volume of a thin shell. The :py:class:`~pisces.geometry.base.RadialCoordinateSystem`
    class implements this as :py:meth:`~pisces.geometry.base.RadialCoordinateSystem.shell_volume`, which takes :math:`r` and
    returns the shell volume. In the case of a pseudo-spherical coordinate system, there is no closed form for this shell_volume in
    terms of elementary functions.

    .. note::

        The expression may be provided in terms of elliptic integrals; however, this is hardly worth the effort in
        a computational context...

    We instead take a different approach. For any pseudo-spherical coordinate system, the Lame coefficients are known and the
    **unscaled** Lame coefficients may be constructed by removing any :math:`r` dependence from each of the original expressions. We denote
    these unscaled Lame coefficients as :math:`\tilde{\lambda}_i`. Now, the shell volume is precisely

    .. math::

        V_{\rm shell}(r_0) = dr \int_{r=r_0} J(r,\theta,\phi) dA,

    but we may (in the case of pseudo-spherical coordinate systems) write

    .. math::

        J(r,\theta,\phi) = r^2 \tilde{J}(\theta,\phi),

    where :math:`\tilde{J}` is the unscaled Jacobian formed by taking the product over all of the unscaled Lame coefficients. Thus,

    .. math::
        V_{\rm shell}(r_0) = r^2 dr \int_{r=r_0} \tilde{J}(\theta,\phi) dA = C r^2 dr,

    where :math:`C` is the so-called *shell-coefficient*. This is a function only of the parameters of the coordinate system and
    therefore can be computed once (at instantiation) via quadrature and then stored for rapid use as needed.

    **Summary**:

    This generalized coordinate system retains the angular components from spherical coordinates but modifies the radial
    norm to accommodate anisotropic scaling. The Lamé coefficients characterize the geometric distortion introduced by
    the scaling factors :math:`\eta_x`, :math:`\eta_y`, and :math:`\eta_z`.
    """
    NDIM = 3
    AXES = ["r", "theta", "phi"]
    PARAMETERS = {"scale_x": 1, "scale_y": 1, "scale_z": 1}
    _SKIP_LAMBDIFICATION = ["jacobian"]

    def __init__(self, scale_x: float = 1, scale_y: float = 1, scale_z: float = 1):
        # BASIC SETUP
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.scales = np.array([scale_x, scale_y, scale_z])

        super().__init__(scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)

        # COMPUTING special attributes
        # For these coordinate systems, we need the flux factor and the shell factor,
        # each of which must be computed by quadrature after symbolic reduction.
        # These are already found symbolically vis-a-vis _derive_symbolic_attributes.
        self.shell_parameter = self._compute_shell_coefficient()
        self.flux_parameter = self._compute_flux_coefficient()

    def _compute_shell_coefficient(self):
        func = self.get_derived_attribute_function("r_shell_element")
        integrand = lambda theta, phi: func(1, theta, phi)

        return (
            2
            * np.pi
            * dblquad(
                integrand, 0, np.pi, lambda x: 0 * x, lambda x: 0 * x + 2 * np.pi
            )[0]
        )

    def _compute_flux_coefficient(self):
        func = self.get_derived_attribute_function("r_surface_element")
        integrand = lambda theta, phi: func(1, theta, phi)

        return (
            2
            * np.pi
            * dblquad(
                integrand, 0, np.pi, lambda x: 0 * x, lambda x: 0 * x + 2 * np.pi
            )[0]
        )

    def _derive_symbolic_attributes(self):
        # PERFORM the standard ones.
        super()._derive_symbolic_attributes()
        self._symbolic_attributes["omega"] = sp.simplify(
            self._omega(*self.SYMBAXES, **self.SYMBPARAMS).subs(self.parameters)
        )
        mylog.debug("\t [COMPLETE] Derived  omega...")

        # COMPUTING the r surface element.
        r_surf_element_symbol = (
            self.get_lame_symbolic("theta") * self.get_lame_symbolic("phi")
        ) / (self.get_lame_symbolic("r") * self.SYMBAXES[0] ** 2)
        r_surf_element_symbol = sp.simplify(r_surf_element_symbol.subs(self.parameters))
        self._symbolic_attributes["r_surface_element"] = sp.simplify(
            r_surf_element_symbol
        )
        mylog.debug("\t [COMPLETE] Derived  r_surface_element...")

        # COMPUTING the shell element.
        shell_element = (
            self.get_derived_attribute_symbolic("jacobian") / self.SYMBAXES[0] ** 2
        )
        shell_element = sp.simplify(shell_element.subs(self.parameters))
        self._symbolic_attributes["r_shell_element"] = sp.simplify(shell_element)
        mylog.debug("\t [COMPLETE] Derived  r_shell_element...")

    @staticmethod
    def _omega(r, theta, phi, scale_x=1, scale_y=1, scale_z=1):
        return sp.sqrt(
            (scale_x * sp.sin(theta) * sp.cos(phi)) ** 2
            + (scale_y * sp.sin(theta) * sp.sin(phi)) ** 2
            + (scale_z * sp.cos(theta)) ** 2
        )

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        # PULL the coordinates out of the coordinate arrays.
        omega = self._eval_der_attr_func("omega", coordinates)
        r, theta, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        st, ct, cp, sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)

        # COMPUTE inversion
        x = r * st * cp / omega
        y = r * st * sp / omega
        z = r * ct / omega
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        xi = np.sqrt(
            (self.scale_x * x) ** 2 + (self.scale_y * y) ** 2 + (self.scale_z * z) ** 2
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return np.stack((xi, theta, phi), axis=-1)

    def lame_0(r, theta, phi, scale_x=1, scale_y=1, scale_z=1):
        _o = sp.sqrt(
            (scale_x * sp.sin(theta) * sp.cos(phi)) ** 2
            + (scale_y * sp.sin(theta) * sp.sin(phi)) ** 2
            + (scale_z * sp.cos(theta)) ** 2
        )
        return 1 / _o

    def lame_1(r, theta, phi, scale_x=1, scale_y=1, scale_z=1):
        _o = sp.sqrt(
            (scale_x * sp.sin(theta) * sp.cos(phi)) ** 2
            + (scale_y * sp.sin(theta) * sp.sin(phi)) ** 2
            + (scale_z * sp.cos(theta)) ** 2
        )

        return r * sp.sqrt((sp.diff(1 / _o, theta) ** 2 + (1 / _o) ** 2))

    def lame_2(r, theta, phi, scale_x=1, scale_y=1, scale_z=1):
        _o = sp.sqrt(
            (scale_x * sp.sin(theta) * sp.cos(phi)) ** 2
            + (scale_y * sp.sin(theta) * sp.sin(phi)) ** 2
            + (scale_z * sp.cos(theta)) ** 2
        )

        return r * sp.sqrt((sp.diff(1 / _o, phi) ** 2 + (sp.sin(theta) / _o) ** 2))

    def jacobian(self, coordinates: NDArray) -> NDArray:
        return np.prod(self.eval_lame(coordinates), axis=-1)

    def integrate_in_shells(
        self, field: Union[np.ndarray, Callable], radii: np.ndarray
    ):
        # Interpolate the integrand
        if callable(field):
            f = lambda r: field(r) * r**2
        else:
            f = InterpolatedUnivariateSpline(radii, field * radii**2)

        return self.shell_parameter * integrate(f, radii, x_0=radii[0], minima=True)

    def solve_radial_poisson_problem(
        self,
        density_profile: Union[Callable, np.ndarray],
        coordinates: np.ndarray,
        /,
        *,
        num_points: int = 1000,
        scale: str = "log",
        psi: Callable = None,
        powerlaw_index: int = None,
    ) -> np.ndarray:
        r"""
        Solve Poisson's equation for ellipsoidal coordinates, given a density profile.

        This method computes the gravitational potential :math:`\Phi(\mathbf{x})` (in Planck units) for a given density profile
        :math:`\rho(r)` in ellipsoidal coordinates by solving Poisson's equation:

        .. math::
            \nabla^2 \Phi = 4 \pi \rho(r).

        In ellipsoidal coordinates, the effective radius :math:`m` is defined as:

        .. math::
            m^2 = \sum_i \eta_i^2 x_i^2,

        where :math:`\eta_i` are the scale parameters of the ellipsoidal coordinate system. This method generalizes
        the spherical case to account for anisotropic mass distributions by incorporating the geometry of the
        ellipsoidal system.

        The solution is returned in Planck units, with natural units of :math:`[M]/[L]`. The density profile should
        be supplied in units of :math:`[M]/[L]^3` and the coordinates in units of :math:`[L]`. This ensures
        compatibility with the computational framework.

        Parameters
        ----------
        density_profile : Union[Callable, np.ndarray]
            The density profile of the system, which can either be:

            - A callable function that takes a single argument (``radius``) and returns the density.
            - A numerical array of density values corresponding to the ``coordinates``.

        coordinates : np.ndarray
            Array of ellipsoidal coordinates where the potential is computed. Must follow the format
            ``(..., NDIM)``, where ``NDIM`` is the number of dimensions.

        num_points : int, optional
            Number of points for integration grids. Default is `1000`.

        scale : str, optional
            Scaling method for grid construction. Can be `'log'` or `'linear'`. Default is `'log'`.

        psi : Callable, optional
            Precomputed :math:`\psi(r)` function for the density profile. If provided, this will be
            used directly. If `None`, it will be computed from the density profile.

        powerlaw_index : int, optional
            Power-law index for the asymptotic density behavior. Required when `density_profile` is
            an array and `psi` is not provided.

        Returns
        -------
        potential : np.ndarray
            Array of computed gravitational potential values at each coordinate, in units of mass/length.

        Raises
        ------
        ValueError
            If the input density profile is not callable and `powerlaw_index` is not specified.
        ValueError
            If the coordinate system's scale parameters cannot be extracted.

        Notes
        -----
        The potential is computed as:

        .. math::

            \Phi(\mathbf{x}) = -\pi \frac{1}{(\prod_i \eta_i)^2} \int_0^\infty
            \frac{\psi(\infty) - \psi(\xi(\tau))}{\sqrt{\prod_i (\tau + \eta_i^{-2})}} \, d\tau,

        where:

        - :math:`\psi(r)` is computed based on the density profile.
        - :math:`\xi(\tau)` is the effective radius for a given integration parameter :math:`\tau`.

        See :ref:`poisson_equation` for details on the relevant theory.

        Examples
        --------

        **Comparison to the Spherical Case**

        As a baseline, consider an :py:class:`OblateHomoeoidalCoordinateSystem` with eccentricity :math:`e = 0.0`, which
        reduces to the spherical case. Using the Hernquist density profile:

        .. math::
            \rho(r) = \frac{\rho_0}{\xi (1 + \xi)^3},

        where :math:`\xi = r/r_s`, and the corresponding potential:

        .. math::

            \Phi(r) = -2\pi\frac{G\rho_0 r_s^3}{r + r_s}.

        We compute and compare the potentials:

        .. plot::
            :include-source:

            >>> from pisces.profiles.density import HernquistDensityProfile
            >>> from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Hernquist profile with rho_0 = 1, r_s = 1
            >>> dp = HernquistDensityProfile(rho_0=1, r_s=1)
            >>> r = np.geomspace(1e-1, 1e3, 1000)
            >>> known_potential = -2 * np.pi / (1 + r)
            >>>
            >>> # Oblate system with e = 0 (spherical)
            >>> cs= OblateHomoeoidalCoordinateSystem(ecc=0.0)
            >>>
            >>> # The coordinates need to be constructed as (..., 3) with all the
            >>> # required coordinates. For this example, we have only 1 phi and 1 theta
            >>> # in the coordinates.
            >>> coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
            >>> computed_potential = cs.solve_radial_poisson_problem(
            ...     dp, coords)
            >>>
            >>> # Plot comparison
            >>> _ = plt.loglog(r, -known_potential, label="True Spherical Potential",color='k',ls='-')
            >>> _ = plt.loglog(r, -computed_potential[:,0,0], label="Computed (e=0)",color='red',ls='--')
            >>> _ = plt.xlabel("Radius (r)")
            >>> _ = plt.ylabel(r"Potential, $-\Phi(r)$")
            >>> _ = plt.legend()
            >>> plt.show()

        **Effect of Ellipticity**

        We can now look at what happens when we have various degrees of eccentricity in the coordinate system. In this
        case, we'll look at :math:`\Phi(r, \theta = 0)` in coordinate systems with differing eccentricities. Note that
        the choice of :math:`\theta` plays a considerable role, :math:`\theta` maximizes the difference from spherical, while
        :math:`\theta = \pi/2` minimizes it.

        .. plot::
            :include-source:

            >>> from pisces.profiles.density import NFWDensityProfile
            >>> from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
            >>> import numpy as np
            >>> import unyt
            >>> from pisces.utilities.physics import G
            >>> import matplotlib.pyplot as plt
            >>>
            >>> # Hernquist profile with rho_0 = 1, r_s = 1
            >>> dp = NFWDensityProfile(rho_0=3.66e6,r_s=25.3)
            >>> r = np.geomspace(1e-1, 1e5, 1000)
            >>>
            >>> # Oblate system with e = 0 (spherical)
            >>> cs0 = OblateHomoeoidalCoordinateSystem(ecc=0.00)
            >>> cs= OblateHomoeoidalCoordinateSystem(ecc=0.86)
            >>>
            >>> # The coordinates need to be constructed as (..., 3) with all the
            >>> # required coordinates. For this example, we have only 1 phi and 1 theta
            >>> # in the coordinates.
            >>> coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
            >>> computed_potential = (unyt.unyt_array(cs.solve_radial_poisson_problem(
            ...     dp, coords)[:,0,0],"Msun/kpc") * G).to_value("km**2/s**2")
            >>> computed_potential0 = (unyt.unyt_array(cs0.solve_radial_poisson_problem(
            ...     dp, coords)[:,0,0],"Msun/kpc") * G).to_value("km**2/s**2")
            >>> # Plot comparison
            >>> _ = plt.semilogx(r, computed_potential, label="True Spherical Potential",color='k',ls='-')
            >>> _ = plt.semilogx(r, computed_potential0, label="True Spherical Potential",ls='-')
            >>> plt.show()


        .. plot::
            :include-source:

           # >>> from pisces.profiles.density import HernquistDensityProfile
           # >>> from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
           # >>> import numpy as np
           # >>> import matplotlib.pyplot as plt
           # >>> from matplotlib.colors import Normalize
           # >>>
           # >>> # Hernquist profile with rho_0 = 1, r_s = 1
           # >>> dp = HernquistDensityProfile(rho_0=1, r_s=1)
           # >>> r = np.geomspace(1e-1, 1e3, 1000)
           # >>> known_potential = -2 * np.pi / (1 + r)
           # >>>
           # >>> # Setup the figure
           # >>> fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6,10))
           # >>> _ = axes[0].loglog(r, -known_potential, label="True Spherical Potential",color='k',ls='-')
           # >>> _ = axes[1].semilogx(r, -known_potential/np.abs(known_potential[0]), color='k',ls='-')
           # >>> _ = axes[1].set_xlabel("Radius (r)")
           # >>> _ = axes[0].set_ylabel(r"Potential $-\Phi(r)$")
           # >>> _ = axes[1].set_ylabel(r"Potential, $-\Phi(r)/\Phi(0)$")
           # >>>
           # >>> # Cycle through various eccentricities and plot each one.
           # >>> eccentricities = [0.0,0.1,0.2,0.5,0.7,0.9,0.99]
           # >>> coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
           # >>> for ecc in eccentricities:
           # ...     cs= OblateHomoeoidalCoordinateSystem(ecc=ecc)
           # ...
           # ...     # The coordinates need to be constructed as (..., 3) with all the
           # ...     # required coordinates. For this example, we have only 1 phi and 1 theta
           # ...     # in the coordinates.
           # ...     computed_potential = cs.solve_radial_poisson_problem(
           # ...         dp, coords)
           # ...
           # ...     # Plot comparison
           # ...     _ = axes[1].semilogx(r, -computed_potential[:,0,0]/(np.abs(computed_potential[0,0,0])),color=plt.cm.cool(ecc),ls='--')
           # ...     _ = axes[0].loglog(r, -computed_potential[:,0,0],color=plt.cm.cool(ecc),ls='--')
           # >>> _ = axes[0].legend()
           # >>> plt.colorbar(plt.cm.ScalarMappable(Normalize(vmin=0,vmax=1),
           # ...     cmap=plt.cm.cool),ax=axes, orientation='horizontal',fraction=0.07, label=r'Eccentricity, $e$')
           # >>> plt.show()

        See Also
        --------
        - :py:func:`compute_ellipsoidal_psi` : Computes the :math:`\psi(r)` function for ellipsoidal potentials.
        - :py:func:`solve_poisson_spherical` : Computes the potential for spherical coordinates.
        """
        from pisces.utilities.math_utils.poisson import solve_poisson_ellipsoidal

        return solve_poisson_ellipsoidal(
            density_profile,
            coordinates,
            self,
            num_points=num_points,
            scale=scale,
            psi=psi,
            powerlaw_index=powerlaw_index,
        )


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding,PyMethodOverriding
class OblateHomoeoidalCoordinateSystem(PseudoSphericalCoordinateSystem):
    r"""
    3-dimensional coordinate system with level surfaces forming concentric homoeoidal
    oblate ellipsoids.

    The :py:class:`OblateSpheroidalCoordinateSystem` is a special case of
    :py:class:`PseudoSphericalCoordinateSystem` for which the scaling factors on the
    :math:`x` and :math:`y` axes are the same and the scaling on :math:`z` is unity. This
    creates an axially symmetric ellipsoidal geometry.

    The :py:class:`OblateSpheroidalCoordinateSystem` is parameterized by an eccentricity parameter
    (``ecc``) such that

    .. math::

        \eta = \sqrt{1-\epsilon^2}.

    In this case, :math:`\Omega` takes the form

    .. math::

        \Omega^2(\theta,\phi) = \eta^2 \sin^2\theta + \cos^2\theta = \cos^2\theta - \epsilon^2 \sin^2\theta.


    Thus, the transformation is

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \frac{r}{\Omega} \sin(\theta) \cos(\phi), \\
                                & \quad y = \frac{r}{\Omega} \sin(\theta) \sin(\phi), \\
                                & \quad z = r \cos(\theta), \\
           \text{From Cartesian:} & \quad r = \sqrt{(1-\epsilon^2)(x^2 + y^2) + z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{d}\right), \\
                                  & \quad \phi = \arctan2(y, x).
       \end{aligned}

    where

    .. math::

        d^2 = x^2 + y^2 + z^2

    Notes
    -----

    The Lame coefficients in this coordinate system are as follows:

    +---------------+------------------------------------------------------------------------------------------+
    | **Axis**      | **Lamé Coefficient**                                                                     |
    +===============+==========================================================================================+
    | :math:`r`     | :math:`\frac{1}{\sqrt{1 - \epsilon^2 \sin^2 \theta}}`                                    |
    +---------------+------------------------------------------------------------------------------------------+
    | :math:`\theta`| :math:`\frac{r}{\sqrt{1 - \epsilon^2 \sin^2 \theta}} \sqrt{1 + \frac{\epsilon^4 \sin^2\  |
    |               | \theta \cos^2 \theta}{(1 - \epsilon^2 \sin^2 \theta)^2}}`                                |
    +---------------+------------------------------------------------------------------------------------------+
    | :math:`\phi`  | :math:`r \sin \theta \cdot \sqrt{1 - \epsilon^2 \sin^2 \theta}`                          |
    +---------------+------------------------------------------------------------------------------------------+


    **Mathematical Background**:

    In the most general case of a :py:class:`PseudoSphericalCoordinateSystem`, the effective
    radius is given by the prescription that

    .. math::

        r^2 = \eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2.

    In spheroidal coordinates, we instead have

    .. math::

        r^2 = \eta_0^2 (x^2+y^2) + z^2

    When :math:`\eta < 1`, this is an oblate spheroid and when :math:`\eta > 1`, it is
    a prolate ellipsoid.

    In the oblate case, we introduce the eccentricity of the coordinate system :math:`\epsilon`,
    such that

    .. math::

        \epsilon^2 = 1-\eta^2,

    and

    .. math::

        r^2 = (1-\epsilon^2)(x^2 + y^2) + z^2

    In this coordinate system,

    .. math::

        \Omega(\phi, \theta) = \sqrt{1-\epsilon^2\sin^2\theta}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = \frac{r}{\Omega(\phi, \theta)} \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \frac{1/\Omega(\phi, \theta)}.

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative. Because
       :math:`\partial_\theta \hat{r} = \hat{\theta}`, we may express this as

       .. math::

            \lambda_\theta = \left| \frac{\partial {\bf r}}{\partial \theta} \right| = \frac{r}{\Omega}\left[1 + \frac{1}{\Omega^2}\left(\frac{\partial \Omega}{\partial \theta}\right)^2\right]^{1/2}

       where

       .. math::

            \frac{\partial \Omega}{\partial \theta} = -\frac{\epsilon^2 \sin \theta \cos \theta}{\Omega}

       .. note::

           In the special case :math:`\eta_x=\eta_y=\eta_z`, :math:`\Omega = 1` and :math:`\partial_\theta \Omega = 0`, so
           the entire right hand term under the radical is zero and we get

           .. math::
               \left|\frac{\partial {\bf r}}{\partial \theta} \right| = r.

    3. :math:`\phi` **Component**:

       The Lamé coefficient for :math:`\phi` involves a similar structure:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \phi}\right| &= r\left|\frac{1}{\Omega} \sin(\theta) \hat{\phi}\right|\\
          &= \frac{r}{\Omega}\sin(\theta)
          \end{aligned}

    Examples
    --------
    The Oblate Homoeoidal coordinate system is visualized with axial slices of constant :math:`\phi = 0`.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem

        Initialize the coordinate system:

        >>> ecc = 0.9  # Eccentricity
        >>> coordinate_system = OblateHomoeoidalCoordinateSystem(ecc)

        Define the coordinate ranges:

        >>> r_vals = np.linspace(0, 2, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, np.pi, 12)  # Angular values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`r` surfaces:

        >>> for r in r_vals:
        ...     theta = np.linspace(0, np.pi, 200)
        ...     coords = np.stack([r * np.ones_like(theta), theta, np.full_like(theta, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        Plot constant :math:`\theta` surfaces:

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 2, 200)
        ...     coords = np.stack([r, theta * np.ones_like(r), np.full_like(r, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        >>> _ = plt.title('Oblate Homoeoidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()

    """
    NDIM = 3
    AXES = ["r", "theta", "phi"]
    PARAMETERS = {"ecc": 0.0}

    def __init__(self, ecc: float = 0.0):
        self.ecc = ecc
        self.scale_x = self.scale_y = np.sqrt(1 - ecc**2)
        self.scale_z = 1
        super(PseudoSphericalCoordinateSystem, self).__init__(ecc=self.ecc)

        # COMPUTING special attributes
        # For these coordinate systems, we need the flux factor and the shell factor,
        # each of which must be computed by quadrature after symbolic reduction.
        # These are already found symbolically vis-a-vis _derive_symbolic_attributes.
        self.shell_parameter = self._compute_shell_coefficient()
        self.flux_parameter = self._compute_flux_coefficient()

    def _compute_shell_coefficient(self):
        func = self.get_derived_attribute_function("r_shell_element")
        integrand = lambda theta: func(1, theta, 0)

        return 2 * np.pi * quad(integrand, 0, np.pi)[0]

    def _compute_flux_coefficient(self):
        func = self.get_derived_attribute_function("r_surface_element")
        integrand = lambda theta: func(1, theta, 0)

        return 2 * np.pi * quad(integrand, 0, np.pi)[0]

    @staticmethod
    def _omega(r, theta, phi, ecc=0.0):
        return sp.sqrt(1 - (ecc * sp.sin(theta)) ** 2)

    def lame_0(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.sin(theta)) ** 2)
        return 1 / _o

    def lame_1(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.sin(theta)) ** 2)

        return r * sp.sqrt((sp.diff(1 / _o, theta) ** 2 + (1 / _o) ** 2))

    def lame_2(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.sin(theta)) ** 2)

        return r * sp.sqrt((sp.diff(1 / _o, phi) ** 2 + (sp.sin(theta) / _o) ** 2))

    def __str__(self):
        return f"<{self.__class__.__name__}(ecc={self.ecc})>"

    def __repr__(self):
        return self.__str__()


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class ProlateHomoeoidalCoordinateSystem(PseudoSphericalCoordinateSystem):
    r"""
    3-dimensional coordinate system with level surfaces forming concentric homoeoidal
    prolate ellipsoids.

    The :py:class:`ProlateSpheroidalCoordinateSystem` is a special case of
    :py:class:`PseudoSphericalCoordinateSystem` for which the scaling factors on the
    :math:`x` and :math:`y` axes are unity. This creates an axially symmetric ellipsoidal geometry.

    The :py:class:`ProlateSpheroidalCoordinateSystem` is parameterized by an eccentricity parameter
    (``ecc``) such that

    .. math::

        \eta = \sqrt{1-\epsilon^2}.

    Thus, the transformation is

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = r \sin(\theta) \cos(\phi), \\
                                & \quad y = r \sin(\theta) \sin(\phi), \\
                                & \quad z = \frac{r \cos(\theta)}{\sqrt{1-\epsilon^2}}, \\
           \text{From Cartesian:} & \quad r = \sqrt{x^2 + y^2 + (1-\epsilon^2)z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{r}\right), \\
                                  & \quad \phi = \arctan2(y, x).
       \end{aligned}

    Notes
    -----

    The Lame coefficients in this coordinate system are as follows:

    +---------------+------------------------------------------------------------------------------------------+
    | **Axis**      | **Lamé Coefficient**                                                                     |
    +===============+==========================================================================================+
    | :math:`r`     | :math:`\sqrt{1 - \epsilon^2 \cos^2 \theta}`                                              |
    +---------------+------------------------------------------------------------------------------------------+
    | :math:`\theta`| :math:`\frac{r}{\sqrt{1 - \epsilon^2 \cos^2 \theta}} \sqrt{1 + \frac{\epsilon^4 \sin^2\  |
    |               | \theta \cos^2 \theta}{(1 - \epsilon^2 \cos^2 \theta)^2}}`                                |
    +---------------+------------------------------------------------------------------------------------------+
    | :math:`\phi`  | :math:`r \sin \theta \cdot \sqrt{1 - \epsilon^2 \cos^2 \theta}`                          |
    +---------------+------------------------------------------------------------------------------------------+


    **Mathematical Background**:

    In the most general case of a :py:class:`PseudoSphericalCoordinateSystem`, the effective
    radius is given by the prescription that

    .. math::

        r^2 = \eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2.

    In spheroidal coordinates, we instead have

    .. math::

        r^2 = x^2+y^2 + \eta_0^2 z^2

    When :math:`\eta < 1`, this is a prolate spheroid and when :math:`\eta > 1`, it is
    an oblate ellipsoid.

    In the prolate case, we introduce the eccentricity of the coordinate system :math:`\epsilon`,
    such that

    .. math::

        \epsilon^2 = 1-\eta^2,

    and

    .. math::

        r^2 = x^2 + y^2 +  (1-\epsilon^2)z^2

    .. math::

        \Omega(\phi, \theta) = \sqrt{1-\epsilon^2\cos^2\theta}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = \frac{r}{\Omega(\phi, \theta)} \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \frac{1/\Omega(\phi, \theta)}.

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative. Because
       :math:`\partial_\theta \hat{r} = \hat{\theta}`, we may express this as

       .. math::

            \lambda_\theta = \left| \frac{\partial {\bf r}}{\partial \theta} \right| = \frac{r}{\Omega}\left[1 + \frac{1}{\Omega^2}\left(\frac{\partial \Omega}{\partial \theta}\right)^2\right]^{1/2}

       where

       .. math::

            \frac{\partial \Omega}{\partial \theta} = \frac{\epsilon^2 \sin \theta \cos \theta}{\Omega}

       .. note::

           In the special case :math:`\eta_x=\eta_y=\eta_z`, :math:`\Omega = 1` and :math:`\partial_\theta \Omega = 0`, so
           the entire right hand term under the radical is zero and we get

           .. math::
               \left|\frac{\partial {\bf r}}{\partial \theta} \right| = r.

    3. :math:`\phi` **Component**:

       The Lamé coefficient for :math:`\phi` involves a similar structure:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \phi}\right| &= r\left|\frac{1}{\Omega} \sin(\theta) \hat{\phi}\right|\\
          &= \frac{r}{\Omega}\sin(\theta)
          \end{aligned}

    Examples
    --------
    The Oblate Homoeoidal coordinate system is visualized with axial slices of constant :math:`\phi = 0`.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import ProlateHomoeoidalCoordinateSystem

        Initialize the coordinate system:

        >>> ecc = 0.9  # Eccentricity
        >>> coordinate_system = ProlateHomoeoidalCoordinateSystem(ecc)

        Define the coordinate ranges:

        >>> r_vals = np.linspace(0, 2, 6)  # Radial distances
        >>> theta_vals = np.linspace(0, np.pi, 12)  # Angular values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`r` surfaces:

        >>> for r in r_vals:
        ...     theta = np.linspace(0, np.pi, 200)
        ...     coords = np.stack([r * np.ones_like(theta), theta, np.full_like(theta, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        Plot constant :math:`\theta` surfaces:

        >>> for theta in theta_vals:
        ...     r = np.linspace(0, 2, 200)
        ...     coords = np.stack([r, theta * np.ones_like(r), np.full_like(r, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        >>> _ = plt.title('Prolate Homoeoidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()

    """
    NDIM = 3
    AXES = ["r", "theta", "phi"]
    PARAMETERS = {"ecc": 0.0}

    def __init__(self, ecc: float = 0.0):
        self.ecc = ecc
        self.scale_x = self.scale_y = 1
        self.scale_z = np.sqrt(1 - ecc**2)
        super(PseudoSphericalCoordinateSystem, self).__init__(ecc=self.ecc)

        # COMPUTING special attributes
        # For these coordinate systems, we need the flux factor and the shell factor,
        # each of which must be computed by quadrature after symbolic reduction.
        # These are already found symbolically vis-a-vis _derive_symbolic_attributes.
        # COMPUTING special attributes
        # For these coordinate systems, we need the flux factor and the shell factor,
        # each of which must be computed by quadrature after symbolic reduction.
        # These are already found symbolically vis-a-vis _derive_symbolic_attributes.
        self.shell_parameter = self._compute_shell_coefficient()
        self.flux_parameter = self._compute_flux_coefficient()

    def _compute_shell_coefficient(self):
        func = self.get_derived_attribute_function("r_shell_element")
        integrand = lambda theta: func(1, theta, 0)

        return 2 * np.pi * quad(integrand, 0, np.pi)[0]

    def _compute_flux_coefficient(self):
        func = self.get_derived_attribute_function("r_surface_element")
        integrand = lambda theta: func(1, theta, 0)

        return 2 * np.pi * quad(integrand, 0, np.pi)[0]

    @staticmethod
    def _omega(r, theta, phi, ecc=0.0):
        return sp.sqrt(1 - (ecc * sp.cos(theta)) ** 2)

    def lame_0(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.cos(theta)) ** 2)
        return 1 / _o

    def lame_1(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.cos(theta)) ** 2)

        return r * sp.sqrt((sp.diff(1 / _o, theta) ** 2 + (1 / _o) ** 2))

    def lame_2(r, theta, phi, ecc=0.0):
        _o = sp.sqrt(1 - (ecc * sp.cos(theta)) ** 2)

        return r * sp.sqrt((sp.diff(1 / _o, phi) ** 2 + (sp.sin(theta) / _o) ** 2))

    def __str__(self):
        return f"<{self.__class__.__name__}(ecc={self.ecc})>"

    def __repr__(self):
        return self.__str__()


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class OblateSpheroidalCoordinateSystem(CoordinateSystem):
    r"""
    3 Dimensional Oblate Spheroidal coordinate system.

    Oblate Spheroidal coordinates are defined using the hyperbolic coordinates :math:`\mu` and :math:`\nu` such that the surfaces of constant :math:`\mu` are oblate spheroids.

    Conversion to Cartesian coordinates is given by:

    .. math::

       \begin{aligned}
           x &= a \cosh(\mu) \cos(\nu) \cos(\phi), \\
           y &= a \cosh(\mu) \cos(\nu) \sin(\phi), \\
           z &= a \sinh(\mu) \sin(\nu).
       \end{aligned}

    Conversion from Cartesian coordinates:

    .. math::

       \begin{aligned}
           \mu &= \operatorname{arccosh}\left(\frac{\sqrt{x^2 + y^2 + z^2 + a^2}}{2a}\right), \\
           \nu &= \arcsin\left(\frac{z}{a \sinh(\mu)}\right), \\
           \phi &= \arctan2(y, x).
       \end{aligned}

    Parameters
    ----------
    a : float
        Semi-major axis defining the scale of the coordinate system.

    Notes
    -----
    The Lamé coefficients in this system are:

    +---------------+---------------------------------------------+
    | Axis          | Lamé Coefficient                            |
    +===============+=============================================+
    | :math:`\mu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\nu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\phi`  | :math:`a \cosh(\mu) \cos(\nu)`              |
    +---------------+---------------------------------------------+

    Examples
    --------
    The Oblate Spheroidal coordinate system is visualized by converting level surfaces to Cartesian coordinates.
    This plot shows an axial slice of constant :math:`\phi = 0`.

    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import OblateSpheroidalCoordinateSystem

        Initialize the coordinate system:

        >>> a = 1.0  # Semi-major axis
        >>> coordinate_system = OblateSpheroidalCoordinateSystem(a)

        Define the coordinate ranges:

        >>> mu_vals = np.linspace(0, 2, 6)  # Range of mu values
        >>> nu_vals = np.linspace(-np.pi / 2, np.pi / 2, 12)  # Range of nu values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`\mu` surfaces:

        >>> for mu in mu_vals:
        ...     nu = np.linspace(-np.pi / 2, np.pi / 2, 200)
        ...     coords = np.stack([mu * np.ones_like(nu), nu, np.full_like(nu, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        Plot constant :math:`\nu` surfaces:

        >>> for nu in nu_vals:
        ...     mu = np.linspace(0, 2, 200)
        ...     coords = np.stack([mu, np.full_like(mu, nu), np.full_like(mu, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        >>> _ = plt.title('Oblate Spheroidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()

    """
    NDIM = 3
    AXES = ["mu", "nu", "phi"]
    PARAMETERS = {"a": 1.0}

    def __init__(self, a: float = 1.0):
        self.a = a
        super().__init__(a=a)

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        mu, nu, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        x = self.a * np.cosh(mu) * np.cos(nu) * np.cos(phi)
        y = self.a * np.cosh(mu) * np.cos(nu) * np.sin(phi)
        z = self.a * np.sinh(mu) * np.sin(nu)
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        rho = np.sqrt(x**2 + y**2)
        d1_2, d2_2 = (rho + self.a) ** 2 + z**2, (rho - self.a) ** 2 + z**2
        mu = np.arccosh((np.sqrt(d1_2) + np.sqrt(d2_2)) / (2 * self.a))
        nu = np.arccos((np.sqrt(d1_2) - np.sqrt(d2_2)) / (2 * self.a))
        phi = np.arctan2(y, x)
        return np.stack((mu, nu, phi), axis=-1)

    def lame_0(mu, nu, phi, a=1.0):
        return a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2)

    def lame_1(mu, nu, phi, a=1.0):
        return a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2)

    def lame_2(mu, nu, phi, a=1.0):
        return a * sp.cosh(mu) * sp.cos(nu)


# noinspection PyMethodMayBeStatic,PyMethodParameters,PyUnresolvedReferences,PyTypeChecker,PyMethodOverriding
class ProlateSpheroidalCoordinateSystem(CoordinateSystem):
    r"""
    3 Dimensional Prolate Spheroidal coordinate system.

    Prolate Spheroidal coordinates are defined using the hyperbolic coordinates :math:`\mu` and :math:`\nu` such that the surfaces of constant :math:`\mu` are prolate spheroids.

    Conversion to Cartesian coordinates is given by:

    .. math::

       \begin{aligned}
           x &= a \sinh(\mu) \sin(\nu) \cos(\phi), \\
           y &= a \sinh(\mu) \sin(\nu) \sin(\phi), \\
           z &= a \cosh(\mu) \cos(\nu).
       \end{aligned}

    Conversion from Cartesian coordinates:

    .. math::

       \begin{aligned}
           \mu &= \operatorname{arccosh}\left(\frac{\sqrt{x^2 + y^2 + z^2 - a^2}}{2a}\right), \\
           \nu &= \arccos\left(\frac{z}{a \cosh(\mu)}\right), \\
           \phi &= \arctan2(y, x).
       \end{aligned}

    Parameters
    ----------
    a : float
        Semi-major axis defining the scale of the coordinate system.

    Notes
    -----
    The Lamé coefficients in this system are:

    +---------------+---------------------------------------------+
    | Axis          | Lamé Coefficient                            |
    +===============+=============================================+
    | :math:`\mu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\nu`   | :math:`a \sqrt{\sinh^2(\mu) + \sin^2(\nu)}` |
    +---------------+---------------------------------------------+
    | :math:`\phi`  | :math:`a \sinh(\mu) \sin(\nu)`              |
    +---------------+---------------------------------------------+

    Examples
    --------
    The Prolate Spheroidal coordinate system is visualized by converting level surfaces to Cartesian coordinates.
    This plot shows an axial slice of constant :math:`\phi = 0`.

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.geometry.coordinate_systems import ProlateSpheroidalCoordinateSystem

        Initialize the coordinate system:

        >>> a = 1.0  # Semi-major axis
        >>> coordinate_system = ProlateSpheroidalCoordinateSystem(a)

        Define the coordinate ranges:

        >>> mu_vals = np.linspace(0, 2, 6)  # Range of mu values
        >>> nu_vals = np.linspace(0, np.pi, 12)  # Range of nu values
        >>> phi = 0  # Fix the azimuthal angle

        Plot constant :math:`\mu` surfaces:

        >>> for mu in mu_vals:
        ...     nu = np.linspace(0, np.pi, 200)
        ...     coords = np.stack([mu * np.ones_like(nu), nu, np.full_like(nu, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        Plot constant :math:`\nu` surfaces:

        >>> for nu in nu_vals:
        ...     mu = np.linspace(0, 2, 200)
        ...     coords = np.stack([mu, np.full_like(mu, nu), np.full_like(mu, phi)], axis=-1)
        ...     cartesian = coordinate_system._convert_native_to_cartesian(coords)
        ...     _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)

        >>> _ = plt.title('Prolate Spheroidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()
    """
    NDIM = 3
    AXES = ["mu", "nu", "phi"]
    PARAMETERS = {"a": 1.0}

    def __init__(self, a: float = 1.0):
        self.a = a
        super().__init__(a=a)

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        mu, nu, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        x = self.a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
        y = self.a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
        z = self.a * np.cosh(mu) * np.cos(nu)
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        rho = np.sqrt(x**2 + y**2)
        d1_2, d2_2 = (rho) ** 2 + (z + self.a) ** 2, (rho) ** 2 + (z - self.a) ** 2
        mu = np.arccosh((np.sqrt(d1_2) + np.sqrt(d2_2)) / (2 * self.a))
        nu = np.arccos((np.sqrt(d1_2) - np.sqrt(d2_2)) / (2 * self.a))
        phi = np.arctan2(y, x)
        return np.stack((mu, nu, phi), axis=-1)

    def lame_0(mu, nu, phi, a=1.0):
        return a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2)

    def lame_1(mu, nu, phi, a=1.0):
        return a * sp.sqrt(sp.sinh(mu) ** 2 + sp.sin(nu) ** 2)

    def lame_2(mu, nu, phi, a=1.0):
        return a * sp.sinh(mu) * sp.sin(nu)
