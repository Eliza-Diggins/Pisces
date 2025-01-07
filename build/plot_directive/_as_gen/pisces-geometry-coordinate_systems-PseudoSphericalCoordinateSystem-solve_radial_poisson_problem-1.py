from pisces.profiles.density import HernquistDensityProfile
from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
import numpy as np
import matplotlib.pyplot as plt
dp = HernquistDensityProfile(rho_0=1, r_s=1)
r = np.geomspace(1e-1, 1e3, 1000)
known_potential = -2 * np.pi / (1 + r)
cs= OblateHomoeoidalCoordinateSystem(ecc=0.0)
coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
computed_potential = cs.solve_radial_poisson_problem(
    dp, coords)
_ = plt.loglog(r, -known_potential, label="True Spherical Potential",color='k',ls='-')
_ = plt.loglog(r, -computed_potential[:,0,0], label="Computed (e=0)",color='red',ls='--')
_ = plt.xlabel("Radius (r)")
_ = plt.ylabel(r"Potential, $-\Phi(r)$")
_ = plt.legend()
plt.show()
