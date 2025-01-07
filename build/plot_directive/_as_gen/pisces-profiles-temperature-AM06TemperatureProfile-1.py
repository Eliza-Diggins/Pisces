import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import AM06TemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = AM06TemperatureProfile(T_0=4.0, a=300.0, a_c=50.0, c=0.2)
T = profile(r)
#
_ = plt.semilogx(r, T, 'g-', label='AM06 Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
