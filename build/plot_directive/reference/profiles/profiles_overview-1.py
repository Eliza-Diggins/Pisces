from pisces.profiles.density import NFWDensityProfile
import matplotlib.pyplot as plt
import numpy as np
nfw_density = NFWDensityProfile(rho_0=1e5,r_s=150)
radii = np.geomspace(1e-2,1e4,1000)
plt.loglog(radii,nfw_density(radii),'k-')
plt.ylabel(r"Density, Msun/kpc^3")
plt.xlabel(r"Radius, kpc")
plt.show()
