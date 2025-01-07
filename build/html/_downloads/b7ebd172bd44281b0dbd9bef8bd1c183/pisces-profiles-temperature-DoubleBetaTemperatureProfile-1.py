import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import DoubleBetaTemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = DoubleBetaTemperatureProfile(
    T_0=5.0, r_c=100.0, beta_1=0.8, T_1=3.0, beta_2=1.2
)
T = profile(r)
#
_ = plt.semilogx(r, T, 'c-', label='Double Beta Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
