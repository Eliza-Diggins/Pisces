r"""
Coordinate Systems for Pisces
=============================

This module contains all of the coordinate systems for the Pisces library. This includes various spheroidal
coordinate systems, cartesian coordinate systems, cylindrical coordinate systems, and others.

Mathematical Background
=======================

An orthogonal coordinate system is defined by a set of coordinate surfaces that intersect
at right angles. Each coordinate axis has an associated Lame coefficient, :math:`h_i`, which
scales differential elements along that axis. Orthogonal systems are useful in physics and
engineering because they allow the calculation of differential operators (e.g., gradient,
divergence, and Laplacian) based on Lame coefficients, simplifying the complexity of vector
calculus in curved spaces.

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
from pisces.geometry.base import CoordinateSystem
from pisces.geometry.utils import lame_coefficient
from numpy.typing import NDArray
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
    """
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

class SphericalCoordinateSystem(CoordinateSystem):
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
