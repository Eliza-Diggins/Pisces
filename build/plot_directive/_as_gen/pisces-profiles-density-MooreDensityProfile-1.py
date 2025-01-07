import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.density import MooreDensityProfile
#
r = np.linspace(0.1, 10, 100)
profile = MooreDensityProfile(rho_0=1.0, r_s=1.0)
rho = profile(r)
#
_ = plt.loglog(r, rho, 'k-', label='Moore Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Density (rho)')
_ = plt.legend()
plt.show()
