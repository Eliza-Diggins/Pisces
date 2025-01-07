import matplotlib.pyplot as plt
from pisces.geometry.coordinate_systems import PolarCoordinateSystem
#
# Initialize the coordinate system:
#
coordinate_system = PolarCoordinateSystem()
#
# Define the radial and angular ranges:
#
r_vals = np.linspace(0, 1, 6)  # Radial distances
theta_vals = np.linspace(0, 2 * np.pi, 12)  # Angular values
#
# Plot concentric circles (constant r):
#
for r in r_vals:
    theta = np.linspace(0, 2 * np.pi, 200)
    coords = np.stack([r * np.ones_like(theta), theta], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)
#
# Plot radial lines (constant theta):
#
for theta in theta_vals:
    r = np.linspace(0, 1, 200)
    coords = np.stack([r, theta * np.ones_like(r)], axis=-1)
    cartesian = coordinate_system._convert_native_to_cartesian(coords)
    _ = plt.plot(cartesian[:, 0], cartesian[:, 1], 'k-', lw=0.5)
#
_ = plt.title('Polar Coordinate System')
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.axis('equal')
plt.show()
