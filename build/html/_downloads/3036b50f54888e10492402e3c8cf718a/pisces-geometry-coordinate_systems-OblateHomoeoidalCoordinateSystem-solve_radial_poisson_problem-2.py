from pisces.profiles.density import NFWDensityProfile
from pisces.geometry.coordinate_systems import OblateHomoeoidalCoordinateSystem
import numpy as np
import unyt
from pisces.utilities.physics import G
import matplotlib.pyplot as plt
dp = NFWDensityProfile(rho_0=3.66e6,r_s=25.3)
r = np.geomspace(1e-1, 1e5, 1000)
cs0 = OblateHomoeoidalCoordinateSystem(ecc=0.00)
cs= OblateHomoeoidalCoordinateSystem(ecc=0.86)
coords = np.moveaxis(np.meshgrid(r,[0],[0],indexing='ij'),0,-1)
computed_potential = (unyt.unyt_array(cs.solve_radial_poisson_problem(
    dp, coords)[:,0,0],"Msun/kpc") * G).to_value("km**2/s**2")
computed_potential0 = (unyt.unyt_array(cs0.solve_radial_poisson_problem(
    dp, coords)[:,0,0],"Msun/kpc") * G).to_value("km**2/s**2")
_ = plt.semilogx(r, computed_potential, label="True Spherical Potential",color='k',ls='-')
_ = plt.semilogx(r, computed_potential0, label="True Spherical Potential",ls='-')
plt.show()
