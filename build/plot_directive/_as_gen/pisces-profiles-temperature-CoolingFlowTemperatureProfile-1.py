import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import CoolingFlowTemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = CoolingFlowTemperatureProfile(T_0=5.0, r_c=100.0, a=0.8)
T = profile(r)
#
_ = plt.semilogx(r, T, 'm-', label='Cooling Flow Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
