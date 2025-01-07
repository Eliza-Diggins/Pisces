import matplotlib.pyplot as plt
from pisces.geometry.coordinate_systems import CartesianCoordinateSystem2D
#
# Initialize the coordinate system:
#
coordinate_system = CartesianCoordinateSystem2D()
#
# Define a grid of points:
#
x_vals = np.linspace(-1, 1, 10)  # x values
y_vals = np.linspace(-1, 1, 10)  # y values
x, y = np.meshgrid(x_vals, y_vals)
#
# Plot the grid:
#
for i in range(len(x_vals)):
    _ = plt.plot(x[:, i], y[:, i], 'k-', lw=0.5)
    _ = plt.plot(x[i, :], y[i, :], 'k-', lw=0.5)
#
_ = plt.title('Cartesian 2D Coordinate System')
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.axis('equal')
plt.show()
