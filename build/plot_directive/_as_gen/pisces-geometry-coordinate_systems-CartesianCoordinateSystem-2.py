from pisces.geometry.coordinate_systems import CartesianCoordinateSystem
coordinate_system = CartesianCoordinateSystem()
grid = np.mgrid[-1:1:100j,-1:1:100j,-1:1:3j]
grid = np.moveaxis(grid,0,-1) # fix the grid ordering to meet our standard
func = lambda x,y: np.cos(y)*np.sin(x*y)
Z = func(grid[...,0],grid[...,1])
gradZ = coordinate_system.compute_gradient(Z,grid)
image_array = gradZ[:,:,1,0].T
plt.imshow(image_array,origin='lower',extent=(-1,1,-1,1),cmap='inferno') # doctest: +SKIP
plt.show()
