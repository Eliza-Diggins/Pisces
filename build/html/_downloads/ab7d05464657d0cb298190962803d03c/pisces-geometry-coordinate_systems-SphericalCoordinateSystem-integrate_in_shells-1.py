import numpy as np
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
cs = SphericalCoordinateSystem()
r = np.linspace(0,1,1000)
density = np.ones_like(r)
#
# Now we can do the computation
#
mass = cs.integrate_in_shells(density, r)
#
# Now, the mass should go as
#
# .. math::
#
#     M(<r) = 4\pi \int_0^r \rho r^2 dr = \frac{4}{3}\pi r^3 = \frac{4}{3}\pi r^3
#
import matplotlib.pyplot as plt
_ = plt.plot(r,mass)
_ = plt.plot(r, (4/3)*np.pi*r**3)
plt.show()
