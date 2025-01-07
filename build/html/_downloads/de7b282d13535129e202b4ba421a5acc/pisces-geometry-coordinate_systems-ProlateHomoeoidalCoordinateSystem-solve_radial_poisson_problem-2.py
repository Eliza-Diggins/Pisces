from pisces.profiles.density import HernquistDensityProfile
from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
dp = HernquistDensityProfile(rho_0=1, r_s=1)
r = np.geomspace(1e-1, 1e3, 1000)
known_potential = -2 * np.pi / (1 + r)
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6,10))
_ = axes[0].loglog(r, -known_potential, label="True Spherical Potential",color='k',ls='-')
_ = axes[1].semilogx(r, -known_potential/np.abs(known_potential[0]), color='k',ls='-')
_ = axes[1].set_xlabel("Radius (r)")
_ = axes[0].set_ylabel(r"Potential $-\Phi(r)$")
_ = axes[1].set_ylabel(r"Potential, $-\Phi(r)/\Phi(0)$")
eccentricities = [0.0,0.1,0.2,0.5,0.7,0.9,0.99]
coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
for ecc in eccentricities:
    cs= OblateHomoeoidalCoordinateSystem(ecc=ecc)

    # The coordinates need to be constructed as (..., 3) with all the
    # required coordinates. For this example, we have only 1 phi and 1 theta
    # in the coordinates.
    computed_potential = cs.solve_radial_poisson_problem(
        dp, coords)

    # Plot comparison
    _ = axes[1].semilogx(r, -computed_potential[:,0,0]/(np.abs(computed_potential[0,0,0])),color=plt.cm.cool(ecc),ls='--')
    _ = axes[0].loglog(r, -computed_potential[:,0,0],color=plt.cm.cool(ecc),ls='--')
_ = axes[0].legend()
plt.colorbar(plt.cm.ScalarMappable(Normalize(vmin=0,vmax=1),
    cmap=plt.cm.cool),ax=axes, orientation='horizontal',fraction=0.07, label=r'Eccentricity, $e$')
plt.show()
