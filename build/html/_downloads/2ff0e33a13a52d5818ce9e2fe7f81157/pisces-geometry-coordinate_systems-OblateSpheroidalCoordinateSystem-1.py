import matplotlib.pyplot as plt
from pisces.geometry.coordinate_systems import OblateSpheroidalCoordinateSystem
#
# Initialize the coordinate system:
#
a = 1.0  # Semi-major axis
coordinate_system = OblateSpheroidalCoordinateSystem(a)
#
# Define the coordinate ranges:
#
mu_vals = np.linspace(0, 2, 6)  # Range of mu values
nu_vals = np.linspace(-np.pi / 2, np.pi / 2, 12)  # Range of nu values
phi = 0  # Fix the azimuthal angle
#
# Plot constant :math:`\mu` surfaces:
#
for mu in mu_vals:
    nu = np.linspace(-np.pi / 2, np.pi / 2, 200)
    coords = np.stack([mu * np.ones_like(nu), nu, np.full_like(nu, phi)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)
#
# Plot constant :math:`\nu` surfaces:
#
for nu in nu_vals:
    mu = np.linspace(0, 2, 200)
    coords = np.stack([mu, np.full_like(mu, nu), np.full_like(mu, phi)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)
#
_ = plt.title('Oblate Spheroidal Coordinate System')
_ = plt.xlabel('x')
_ = plt.ylabel('z')
_ = plt.axis('equal')
plt.show()
