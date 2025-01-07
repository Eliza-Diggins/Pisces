import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import IsothermalTemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = IsothermalTemperatureProfile(T_0=5.0)
T = profile(r)
#
_ = plt.semilogx(r, T, 'c-', label='Isothermal Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
