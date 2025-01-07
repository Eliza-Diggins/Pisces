# Let's begin by initializing the Cartesian coordinate system:
#
from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
coordinate_system = CartesianCoordinateSystem()
#
# We can now initialize a Cartesian grid. We'll use a slice in X-Y:
#
grid = np.mgrid[-1:1:100j,-1:1:100j,-1:1:3j]
grid = np.moveaxis(grid,0,-1) # fix the grid ordering to meet our standard
#
# Let's now create a function on this geometry.
#
func = lambda x,y: np.cos(y)*np.sin(x*y)
Z = func(grid[...,0],grid[...,1])
#
import matplotlib.pyplot as plt
image_array = Z[:,:,1].T
plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP
