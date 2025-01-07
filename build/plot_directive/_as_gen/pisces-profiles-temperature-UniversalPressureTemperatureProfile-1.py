import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import UniversalPressureTemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = UniversalPressureTemperatureProfile(T_0=5.0, r_s=300.0)
T = profile(r)
#
_ = plt.semilogx(r, T, 'r-', label='Universal Pressure Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
