import matplotlib.pyplot as plt
from pisces.geometry.coordinate_systems import CylindricalCoordinateSystem
#
# Initialize the coordinate system:
#
coordinate_system = CylindricalCoordinateSystem()
#
# Define the radial and angular ranges:
#
rho_vals = np.linspace(0, 1, 6)  # Radial distances
phi_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values
z = 0  # Fix the z-coordinate
#
# Plot concentric circles (constant rho):
#
for rho in rho_vals:
    phi = np.linspace(0, 2 * np.pi, 200)
    coords = np.stack([rho * np.ones_like(phi), phi, np.full_like(phi, z)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)
#
# Plot radial lines (constant phi):
#
for phi in phi_vals:
    rho = np.linspace(0, 1, 200)
    coords = np.stack([rho, phi * np.ones_like(rho), np.full_like(rho, z)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)
#
_ = plt.title('Cylindrical Coordinate System (Slice)')
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.axis('equal')
plt.show()
