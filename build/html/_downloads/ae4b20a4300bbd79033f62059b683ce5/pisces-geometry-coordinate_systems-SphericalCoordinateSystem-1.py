import matplotlib.pyplot as plt
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
#
# Initialize the coordinate system:
#
coordinate_system = SphericalCoordinateSystem()
#
# Define radial and angular ranges:
#
r_vals = np.linspace(0, 1, 6)  # Radial distances
theta_vals = np.linspace(0, np.pi, 12)  # Angular values
phi = 0  # Fix the azimuthal angle
#
# Plot circles (constant r):
#
for r in r_vals:
    theta = np.linspace(0, np.pi, 200)
    coords = np.stack([r * np.ones_like(theta), theta, np.full_like(theta, phi)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)
#
# Plot radial lines (constant theta):
#
for theta in theta_vals:
    r = np.linspace(0, 1, 200)
    coords = np.stack([r, theta * np.ones_like(r), np.full_like(r, phi)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 2], 'k-', lw=0.5)
#
_ = plt.title('Spherical Coordinate System (Slice)')
_ = plt.xlabel('x')
_ = plt.ylabel('z')
_ = plt.axis('equal')
plt.show()
