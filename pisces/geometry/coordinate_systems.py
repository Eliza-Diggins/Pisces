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
from pisces.geometry.base import CoordinateSystem, RadialCoordinateSystem
from pisces.geometry.utils import lame_coefficient
from numpy.typing import NDArray
from scipy.integrate import quad, dblquad
import numpy as np

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
        >>> gradZ = coordinate_system.gradient(grid,Z)
        >>> image_array = gradZ[:,:,1,0].T
        >>> plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP
        >>> plt.show()

    """
    NDIM = 3
    AXES = ['x', 'y', 'z']

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Cartesian is already in native form

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Cartesian is already in native form

    @lame_coefficient(0,axes=[])
    def lame_0(self,coordinates):
        return np.ones_like(coordinates[...,0])

    @lame_coefficient(1,axes=[])
    def lame_1(self,coordinates):
        return np.ones_like(coordinates[...,0])

    @lame_coefficient(2,axes=[])
    def lame_2(self,coordinates):
        return np.ones_like(coordinates[...,0])

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
    AXES = ['x']

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    @lame_coefficient(0, axes=[])
    def lame_0(self, coordinates):
        return np.ones_like(coordinates[..., 0])

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
    AXES = ['x','y']

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        return coordinates  # Already Cartesian

    @lame_coefficient(0, axes=[])
    def lame_0(self, coordinates):
        return np.ones_like(coordinates[..., 0])

    @lame_coefficient(1, axes=[])
    def lame_1(self, coordinates):
        return np.ones_like(coordinates[..., 1])

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

    # noinspection PyMethodMayBeStatic
    def shell_volume(self,radii: NDArray[np.floating]):
        return 4*np.pi*radii**2

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
    AXES = ['r', 'theta']

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

    @lame_coefficient(0,axes=[])
    def lame_0(self,coordinates):
        return np.ones_like(coordinates[...,0])

    @lame_coefficient(1,axes=[0])
    def lame_1(self,coordinates):
        return coordinates[...,0]

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
    | :math:`z`        | :math:`1`                 |
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
    AXES = ['rho', 'phi', 'z']

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

    @lame_coefficient(0,axes=[])
    def lame_0(self,coordinates):
        return np.ones_like(coordinates[...,0])

    @lame_coefficient(1,axes=[0])
    def lame_1(self,coordinates):
        return coordinates[...,0]

    @lame_coefficient(2,axes=[])
    def lame_2(self,coordinates):
        return np.ones_like(coordinates[...,0])

# noinspection DuplicatedCode
class PseudoSphericalCoordinateSystem(RadialCoordinateSystem):
    r"""
    A generalized coordinate system that replaces the standard Euclidean norm with
    a scaled norm of the form:

    .. math::

        r^2 = \eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2.

    The angular coordinates remain the same as in spherical coordinates; however, the
    :math:`r` iso-surfaces are now ellipsoids instead of spheres.

    **Conversion to and from Cartesian coordinates**:

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \frac{r}{\eta_x} \sin(\theta) \cos(\phi), \\
                                & \quad y = \frac{r}{\eta_y} \sin(\theta) \sin(\phi), \\
                                & \quad z = \frac{r}{\eta_z} \cos(\theta), \\
           \text{From Cartesian:} & \quad r = \sqrt{\eta_x^2 x^2 + \eta_y^2 y^2 + \eta_z^2 z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{r}\right), \\
                                  & \quad \phi = \arctan2(y, x).
       \end{aligned}

    **Mathematical Background**:

    For a point :math:`(r, \phi, \theta)` in the pseudo-spherical coordinate system, its Cartesian distance is:

    .. math::

        d^2 = x^2 + y^2 + z^2 = r^2 \left[\frac{\cos^2\phi \sin^2\theta}{\eta_x^2}
        + \frac{\sin^2\phi \sin^2\theta}{\eta_y^2} + \frac{\cos^2\theta}{\eta_z^2}\right].

    We define the *angular scale* function:

    .. math::

        d^2 = r^2 \Omega(\phi, \theta)^2,

    where:

    .. math::

        \Omega(\phi, \theta) = \sqrt{\frac{\cos^2\phi \sin^2\theta}{\eta_x^2}
        + \frac{\sin^2\phi \sin^2\theta}{\eta_y^2} + \frac{\cos^2\theta}{\eta_z^2}}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = r \Omega(\phi, \theta) \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \Omega(\phi, \theta).

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative:

       .. math::

           \begin{aligned}
            \left| \frac{\partial {\bf r}}{\partial \theta} \right| &=
           r \Omega\sqrt{\frac{\sin^2\theta \cos^2\theta}{\Omega^4} \left(
           \frac{\cos^2\phi}{\eta_x^2} + \frac{\sin^2\phi}{\eta_y^2} - \frac{1}{\eta_z^2}
           \right)^2 + 1}, \\
           \text{where} \quad \partial_\theta \Omega &= \frac{\cos\theta \sin\theta}{\Omega}
           \left[\left(\frac{\cos^2\phi}{\eta_x^2} + \frac{\sin^2\phi}{\eta_y^2}\right)
           - \frac{1}{\eta_z^2}\right].
           \end{aligned}

       .. note::

           In the special case :math:`\eta_x=\eta_y=\eta_z`, :math:`\Omega = 1` and :math:`\partial_\theta \Omega = 0`, so
           the entire left hand term under the radical is zero and we get

           .. math::
               \left|\frac{\partial {\bf r}}{\partial \theta} \right| = r.

       We further note that the lame coefficient is precisely this scale factor.

    3. :math:`\phi` **Component**:

       The Lamé coefficient for :math:`\phi` involves a similar structure:

       .. math::

            \left| \frac{\partial {\bf r}}{\partial \phi} \right| =
           r \Omega \sin \theta \sqrt{\frac{\sin^2\theta\cos^2 \phi \sin^2 \phi}{\Omega^4} \left(
           \frac{1}{\eta_y^2} - \frac{1}{\eta_x^2}
           \right)^2 + 1}

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
    AXES = ['r', 'phi', 'theta']

    def __init__(self, scale_x: float,scale_y: float,scale_z: float):
        # BASIC SETUP
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.scales = np.array([scale_x, scale_y, scale_z])

        super().__init__(scale_x,scale_y,scale_z)


    def omega(self, coordinates: NDArray) -> NDArray:
        theta,phi = coordinates[..., 1], coordinates[..., 2]
        st,ct,cp,sp = np.sin(theta),np.cos(theta),np.cos(phi),np.sin(phi)

        return np.sqrt((self.scale_x * st * cp) ** 2 + (self.scale_y * st * sp) ** 2 + (self.scale_z * ct) ** 2)

    # noinspection DuplicatedCode
    def domega_dtheta(self, coordinates: NDArray) -> NDArray:
        theta, phi = coordinates[..., 1], coordinates[..., 2]
        st, ct, cp, sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)

        # Compute Omega
        omega = self.omega(coordinates)

        # Compute the derivative
        factor = cp ** 2 / self.scale_x ** 2 + sp ** 2 / self.scale_y ** 2 - 1 / self.scale_z ** 2
        return (ct * st / omega) * factor

    def domega_dphi(self, coordinates: NDArray) -> NDArray:
        theta, phi = coordinates[..., 1], coordinates[..., 2]
        st, ct, cp, sp = np.sin(theta), np.cos(theta), np.cos(phi), np.sin(phi)

        # Compute Omega
        omega = self.omega(coordinates)

        # Compute the derivative
        factor = 1 / self.scale_y ** 2 - 1 / self.scale_x ** 2
        return (cp * sp * st ** 2 / omega) * factor

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        # PULL the coordinates out of the coordinate arrays.
        r, theta, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]

        # COMPUTE the radial scales
        st,ct,cp,sp = np.sin(theta),np.cos(theta),np.cos(phi),np.sin(phi)

        # COMPUTE inversion
        x = r * st * cp / self.scale_x
        y = r * st * sp / self.scale_y
        z = r * ct / self.scale_z
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        xi = np.sqrt((self.scale_x*x)**2 + (self.scale_y*y)**2 + (self.scale_z*z)**2)
        theta = np.arccos(z / xi)
        phi = np.arctan2(y, x)
        return np.stack((xi, theta, phi), axis=-1)

    def _lame_0_unscaled(self, coordinates) -> NDArray:
        return self.omega(coordinates)

    def _lame_1_unscaled(self, coordinates) -> NDArray:
        r"""
        Compute the unscaled Lamé coefficient for the theta component.

        This is given by:

        .. math::
            \lambda_\theta = \Omega(\phi, \theta) \cdot \sqrt{\Omega^2 + \left(\frac{\partial \Omega}{\partial \theta}\right)^2}.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates with shape (..., 3).

        Returns
        -------
        NDArray
            Unscaled Lamé coefficient for the theta component.
        """
        omega = self.omega(coordinates)
        domega_dtheta = self.domega_dtheta(coordinates)
        return omega * np.sqrt(omega ** 2 + domega_dtheta ** 2)

    def _lame_2_unscaled(self, coordinates) -> NDArray:
        r"""
        Compute the unscaled Lamé coefficient for the phi component.

        This is given by:

        .. math::
            \lambda_\phi = \sin(\theta) \cdot \Omega(\phi, \theta)
                           \cdot \sqrt{\Omega^2 + \left(\frac{\partial \Omega}{\partial \phi}\right)^2}.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates with shape (..., 3).

        Returns
        -------
        NDArray
            Unscaled Lamé coefficient for the phi component.
        """
        theta = coordinates[..., 1]
        omega = self.omega(coordinates)
        domega_dphi = self.domega_dphi(coordinates)
        return np.sin(theta) * omega * np.sqrt(omega ** 2 + domega_dphi ** 2)

    def _unscaled_radial_jacobian(self, coordinates: NDArray) -> NDArray:
        r"""
        Compute the unscaled Jacobian at the specified coordinates.

        The unscaled Jacobian is given by the product of the Lamé coefficients:

        .. math::
            J_{\text{unscaled}} = \lambda_r \cdot \lambda_\theta \cdot \lambda_\phi.

        Parameters
        ----------
        coordinates : NDArray
            Array of coordinates of shape (..., 3), where the final axis contains (r, theta, phi).

        Returns
        -------
        NDArray
            The unscaled Jacobian at the specified points.
        """
        lam_r = self._lame_0_unscaled(coordinates)
        lam_theta = self._lame_1_unscaled(coordinates)
        lam_phi = self._lame_2_unscaled(coordinates)

        return lam_r * lam_theta * lam_phi

    def compute_shell_coefficient(self):
        r"""
        Compute the integral of the unscaled Jacobian over angular coordinates :math:`(\phi, \theta)`.

        Mathematically, this computes:

        .. math::
        sy
            C = \int_0^\pi d\theta \int_0^{2\pi} d\phi J_{\text{unscaled}}(0, \phi, \theta)

        where :math:`J_{\text{unscaled}}` is the product of the unscaled Lamé coefficients:

        .. math::
            J_{\text{unscaled}} = \tilde{\lambda}_r \cdot \tilde{\lambda}_\theta \cdot \tilde{\lambda}_\phi.

        Returns
        -------
        float
            The computed shell coefficient, which is used as a scaling factor for the shell volume.
        """

        def integrand(theta, phi):
            coordinates = np.array([[0, theta, phi]])  # Single row for (r=0, theta, phi)
            return self._unscaled_radial_jacobian(coordinates)

        # Perform the integration over phi [0, 2π] and theta [0, π]
        result, _ = dblquad(
            integrand,  # Function to integrate
            0, 2 * np.pi,  # Limits for phi
            lambda _: 0,  # Lower limit for theta
            lambda _: np.pi  # Upper limit for theta
        )
        return result

    def shell_volume(self, radii: NDArray[np.floating]) -> NDArray[np.floating]:
        return self._shell_coefficient * radii ** 2

    @lame_coefficient(0,axes=[1,2,3])
    def lame_0(self,coordinates):
        return self._lame_0_unscaled(coordinates)

    @lame_coefficient(1,axes=[1,2,3])
    def lame_1(self,coordinates):
        r = coordinates[..., 0]
        return r*self._lame_1_unscaled(coordinates)

    @lame_coefficient(2,axes=[1,2,3])
    def lame_2(self,coordinates):
        r = coordinates[..., 0]
        return r * self._lame_2_unscaled(coordinates)

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

    Thus, the transformation is

    .. math::

       \begin{aligned}
           \text{To Cartesian:} & \quad x = \frac{r}{\sqrt{1-\epsilon^2}} \sin(\theta) \cos(\phi), \\
                                & \quad y = \frac{r}{\sqrt{1-\epsilon^2}} \sin(\theta) \sin(\phi), \\
                                & \quad z = r \cos(\theta), \\
           \text{From Cartesian:} & \quad r = \sqrt{(1-\epsilon^2)(x^2 + y^2) + z^2}, \\
                                  & \quad \theta = \arccos\left(\frac{z}{r}\right), \\
                                  & \quad \phi = \arctan2(y, x).
       \end{aligned}

    Notes
    -----

    The Lame coefficients in this coordinate system are as follows:

    +---------------+-------------------------------------------------------------------------------------+
    | Axis          | Lame Coefficient                                                                    |
    +===============+=====================================================================================+
    | :math:`r`     | :math:`\sqrt{1-\epsilon^2\sin^2\theta}`                                             |
    +---------------+-------------------------------------------------------------------------------------+
    | :math:`\theta`| :math:`r\left[\epsilon^2\left(\epsilon^2 -2\right)\sin^2\theta + 1\right]^{1/2}`    |
    +---------------+-------------------------------------------------------------------------------------+
    | :math:`\phi`  | :math:`r\sin\theta \sqrt{1-\epsilon^2\sin^2\theta}`                                 |
    +---------------+-------------------------------------------------------------------------------------+

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

    For a point :math:`(r, \phi, \theta)` in the pseudo-spherical coordinate system, its Cartesian distance is:

    .. math::

        d^2 = x^2 + y^2 + z^2 = r^2 \left[(1-\epsilon^2)\sin^2\theta + \cos^2 \theta\right] = r^2\left[1-\epsilon^2\sin^2\theta\right].

    Simplifying

    .. math::

        d^2 = r^2\left(1-\epsilon^2\sin^2 \theta\right)

    We define the *angular scale* function:

    .. math::

        d^2 = r^2 \Omega(\phi, \theta)^2,

    where:

    .. math::

        \Omega(\phi, \theta) = \sqrt{1-\epsilon^2\sin^2\theta}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = r \Omega(\phi, \theta) \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \Omega(\phi, \theta).

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \theta}\right| &= r\left|\frac{\partial \Omega}{\partial \theta} \hat{r} + \Omega \hat{\theta}\right|\\
          &= r\left[\epsilon^2\left(\epsilon^2 -2\right)\sin^2\theta + 1\right]^{1/2}
          \end{aligned}

    3. :math:`\phi` **Component**:

       The Lamé coefficient for :math:`\phi` involves a similar structure:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \phi}\right| &= r\left|\Omega \sin(\theta) \hat{\phi}\right|\\
          &= r\Omega \sin(\theta)
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
    AXES = ['r', 'phi', 'theta']

    def __init__(self, ecc=0.0):
        """
        Initialize the Oblate Homoeoidal coordinate system.

        Parameters
        ----------
        ecc: float
            The eccentricity of the coordinate system. This should be a value between 0 and 1.
            Values of ``0`` correspond to spherical coordinates.

        """
        # Perform the standard initialization for the coordinate system.
        self.ecc = ecc
        self.scale_x = self.scale_y = np.sqrt(1 - ecc ** 2)
        self.scale_z = 1
        super(PseudoSphericalCoordinateSystem,self).__init__(ecc=self.ecc)

        # GENERATE the shell volume constant
        self._shell_coefficient = self.compute_shell_coefficient()

    def compute_shell_coefficient(self):
        r"""
        Compute the shell parameter for the coordinate system.

        The shell parameter is an integral that captures the scaling behavior
        of the Lamé coefficients across the angular coordinate, which is necessary
        for determining the volume of a spherical shell.

        For the given homoeoidal coordinate system, the integral is defined as:

        .. math::

            S = \int_0^\pi h_0(\theta) h_1(\theta) h_2(\theta) \sin(\theta) d\theta

        Where:
            - :math:`h_0(\theta)` is the radial Lamé coefficient,
            - :math:`h_1(\theta)` is the angular Lamé coefficient in the polar direction,
            - :math:`h_2(\theta)` is the azimuthal Lamé coefficient.

        Returns
        -------
        float
            The computed shell parameter for this coordinate system.
        """
        # Compute the relevant coefficients
        coef = self.ecc ** 2 * (self.ecc ** 2 - 2)
        lame_0_comp = lambda theta: np.sqrt(1 - (self.ecc * np.sin(theta) ** 2))
        lame_1_comp = lambda theta: np.sqrt(coef * np.sin(theta) ** 2 + 1)
        lame_2_comp = lambda theta: np.sin(theta) * lame_1_comp(theta)

        integrand = lambda theta: lame_0_comp(theta) * lame_2_comp(theta) * lame_1_comp(theta)

        # Perform the integration
        return 2*np.pi*quad(integrand, 0, np.pi)[0]

    def omega(self, coordinates: NDArray) -> NDArray:
        theta = coordinates[..., 1]
        return np.sqrt(1 - (self.ecc*np.sin(theta)**2))


    @lame_coefficient(0,axes=[1])
    def lame_0(self,coordinates):
        return self.omega(coordinates)

    @lame_coefficient(1,axes=[0,1])
    def lame_1(self,coordinates):
        r,theta = coordinates[..., 0], coordinates[..., 1]
        st = np.sin(theta)

        coef = self.ecc**2 * (self.ecc**2 - 2)

        return r*np.sqrt(coef*st**2 + 1)

    @lame_coefficient(2,axes=[0,1])
    def lame_2(self,coordinates):
        r, theta = coordinates[..., 0], coordinates[..., 1]
        omega = self.omega(coordinates)

        return r *omega*np.sin(theta)

    def __str__(self):
        return f"<{self.__class__.__name__}(ecc={self.ecc})>"

    def __repr__(self):
        return self.__str__()

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

    +---------------+-------------------------------------------------------------------------------------+
    | Axis          | Lame Coefficient                                                                    |
    +===============+=====================================================================================+
    | :math:`r`     | :math:`\sqrt{1-\epsilon^2\cos^2\theta}`                                             |
    +---------------+-------------------------------------------------------------------------------------+
    | :math:`\theta`| :math:`r\left[\epsilon^2\left(\epsilon^2 -2\right)\cos^2\theta + 1\right]^{1/2}`    |
    +---------------+-------------------------------------------------------------------------------------+
    | :math:`\phi`  | :math:`r\sin\theta \sqrt{1-\epsilon^2\cos^2\theta}`                                 |
    +---------------+-------------------------------------------------------------------------------------+

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

    For a point :math:`(r, \phi, \theta)` in the pseudo-spherical coordinate system, its Cartesian distance is:

    .. math::

        d^2 = x^2 + y^2 + z^2 = r^2 \left[\sin^2\theta + (1-\epsilon^2)\cos^2 \theta\right] = r^2\left[1-\epsilon^2\cos^2\theta\right].

    Simplifying

    .. math::

        d^2 = r^2\left(1-\epsilon^2\cos^2 \theta\right)

    We define the *angular scale* function:

    .. math::

        d^2 = r^2 \Omega(\phi, \theta)^2,

    where:

    .. math::

        \Omega(\phi, \theta) = \sqrt{1-\epsilon^2\cos^2\theta}.

    **Lamé Coefficients**:

    The position vector is:

    .. math::

        {\bf r} = r \Omega(\phi, \theta) \hat{r}.

    1. **Radial Component**:
       The radial Lamé coefficient is:

       .. math::

           \lambda_r = \left|\frac{\partial {\bf r}}{\partial r}\right| = \Omega(\phi, \theta).

    2. :math:`\theta` **Component**:
       The Lamé coefficient for :math:`\theta` is more complex and involves both :math:`\Omega` and its derivative:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \theta}\right| &= r\left|\frac{\partial \Omega}{\partial \theta} \hat{r} + \Omega \hat{\theta}\right|\\
          &= r\left[\epsilon^2\left(\epsilon^2 -2\right)\cos^2\theta + 1\right]^{1/2}
          \end{aligned}

    3. :math:`\phi` **Component**:

       The Lamé coefficient for :math:`\phi` involves a similar structure:

       .. math::

          \begin{aligned}
          \left|\frac{\partial {\bf r}}{\partial \phi}\right| &= r\left|\Omega \sin(\theta) \hat{\phi}\right|\\
          &= r\Omega \sin(\theta)
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

        >>> _ = plt.title('Oblate Homoeoidal Coordinate System')
        >>> _ = plt.xlabel('x')
        >>> _ = plt.ylabel('z')
        >>> _ = plt.axis('equal')
        >>> plt.show()

    """
    NDIM = 3
    AXES = ['r', 'phi', 'theta']

    def __init__(self, ecc=0.0):
        self.ecc = ecc
        self.scale_x = self.scale_y = 1
        self.scale_z = np.sqrt(1 - ecc ** 2)
        super(PseudoSphericalCoordinateSystem, self).__init__(ecc=self.ecc)

        # GENERATE the shell volume constant
        self._shell_coefficient = self.compute_shell_coefficient()

    def compute_shell_coefficient(self):
        r"""
        Compute the shell parameter for the coordinate system.

        The shell parameter is an integral that captures the scaling behavior
        of the Lamé coefficients across the angular coordinate, which is necessary
        for determining the volume of a spherical shell.

        For the given homoeoidal coordinate system, the integral is defined as:

        .. math::

            S = \int_0^\pi h_0(\theta) h_1(\theta) h_2(\theta) \sin(\theta) d\theta

        Where:
            - :math:`h_0(\theta)` is the radial Lamé coefficient,
            - :math:`h_1(\theta)` is the angular Lamé coefficient in the polar direction,
            - :math:`h_2(\theta)` is the azimuthal Lamé coefficient,

        and each of the Lame coefficients has had any :math:`r` dependence removed.


        Returns
        -------
        float
            The computed shell parameter for this coordinate system.
        """
        # Compute the relevant coefficients
        coef = self.ecc ** 2 * (self.ecc ** 2 - 2)
        lame_0_comp = lambda theta: np.sqrt(1 - (self.ecc * np.cos(theta) ** 2))
        lame_1_comp = lambda theta: np.sqrt(coef * np.cos(theta) ** 2 + 1)
        lame_2_comp = lambda theta: np.sin(theta)*lame_1_comp(theta)

        integrand = lambda theta: lame_0_comp(theta) * lame_2_comp(theta) * lame_1_comp(theta)

        # Perform the integration
        return 2*np.pi*quad(integrand,0,np.pi)[0]

    def omega(self, coordinates: NDArray) -> NDArray:
        theta = coordinates[..., 1]
        return np.sqrt(1 - (self.ecc * np.cos(theta) ** 2))

    @lame_coefficient(0, axes=[1])
    def lame_0(self, coordinates):
        return self.omega(coordinates)

    @lame_coefficient(1, axes=[0, 1])
    def lame_1(self, coordinates):
        r, theta = coordinates[..., 0], coordinates[..., 1]
        ct = np.cos(theta)

        coef = self.ecc ** 2 * (self.ecc ** 2 - 2)

        return r * np.sqrt(coef * ct ** 2 + 1)

    @lame_coefficient(2, axes=[0, 1])
    def lame_2(self, coordinates):
        r, theta = coordinates[..., 0], coordinates[..., 1]
        omega = self.omega(coordinates)

        return r * omega * np.sin(theta)

    def __str__(self):
        return f"<{self.__class__.__name__}(ecc={self.ecc})>"

    def __repr__(self):
        return self.__str__()

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
    AXES = ['mu', 'nu', 'phi']

    def __init__(self, a: float):
        self.a = a
        super().__init__(a)

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        mu, nu, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        x = self.a * np.cosh(mu) * np.cos(nu) * np.cos(phi)
        y = self.a * np.cosh(mu) * np.cos(nu) * np.sin(phi)
        z = self.a * np.sinh(mu) * np.sin(nu)
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        r2 = x ** 2 + y ** 2 + z ** 2
        mu = np.arccosh(np.sqrt(r2 + self.a ** 2) / (2 * self.a))
        nu = np.arcsin(z / (self.a * np.sinh(mu)))
        phi = np.arctan2(y, x)
        return np.stack((mu, nu, phi), axis=-1)

    @lame_coefficient(0, axes=[0,1])
    def lame_0(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.sqrt(np.sinh(mu) ** 2 + np.sin(nu) ** 2)

    @lame_coefficient(1, axes=[0,1])
    def lame_1(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.sqrt(np.sinh(mu) ** 2 + np.sin(nu) ** 2)

    @lame_coefficient(2, axes=[0, 1])
    def lame_2(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.cosh(mu) * np.cos(nu)

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
    AXES = ['mu', 'nu', 'phi']

    def __init__(self, a: float):
        self.a = a
        super().__init__(a)

    def _convert_native_to_cartesian(self, coordinates: NDArray) -> NDArray:
        mu, nu, phi = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        x = self.a * np.sinh(mu) * np.sin(nu) * np.cos(phi)
        y = self.a * np.sinh(mu) * np.sin(nu) * np.sin(phi)
        z = self.a * np.cosh(mu) * np.cos(nu)
        return np.stack((x, y, z), axis=-1)

    def _convert_cartesian_to_native(self, coordinates: NDArray) -> NDArray:
        x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        r2 = x ** 2 + y ** 2 + z ** 2
        mu = np.arccosh(np.sqrt(r2 - self.a ** 2) / (2 * self.a))
        nu = np.arccos(z / (self.a * np.cosh(mu)))
        phi = np.arctan2(y, x)
        return np.stack((mu, nu, phi), axis=-1)

    @lame_coefficient(0, axes=[0,1])
    def lame_0(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.sqrt(np.sinh(mu) ** 2 + np.sin(nu) ** 2)

    @lame_coefficient(1, axes=[0,1])
    def lame_1(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.sqrt(np.sinh(mu) ** 2 + np.sin(nu) ** 2)

    @lame_coefficient(2, axes=[0,1])
    def lame_2(self, coordinates):
        mu, nu = coordinates[..., 0], coordinates[..., 1]
        return self.a * np.sinh(mu) * np.sin(nu)

if __name__ == '__main__':
    h = PseudoSphericalCoordinateSystem(1,1,1)
    print(h.compute_shell_coefficient()-(4*np.pi))