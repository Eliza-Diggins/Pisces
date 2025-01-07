from pisces.profiles.density import HernquistDensityProfile
import numpy as np
dp = HernquistDensityProfile(rho_0=1,r_s=1)
r = np.geomspace(1e-1,1e3,10000)
known_potential = -2*np.pi/(1+r)
#
# Let's now initialize the geometry and compute the solution.
#
from pisces.geometry.coordinate_systems import SphericalCoordinateSystem
cs = SphericalCoordinateSystem()
computed_potential = cs.solve_radial_poisson_problem(dp, r)
#
# We can now make a plot:
#
import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw=dict(hspace=0,height_ratios=[1,0.25]))
_ = axes[0].loglog(r,-known_potential,label='True')
_ = axes[0].loglog(r,-computed_potential, label='Computed')
_ = axes[1].loglog(r, np.abs((known_potential-computed_potential)/known_potential))
_ = axes[0].set_ylabel(r"Potential $\left(-\Phi(r)\right)$")
_ = axes[1].set_ylabel(r"Rel. Err.")
_ = axes[1].set_xlabel(r"Radius")
_ = axes[0].legend()
plt.show()
