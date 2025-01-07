import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.temperature import VikhlininTemperatureProfile
#
r = np.logspace(-1, 2, 100)
profile = VikhlininTemperatureProfile(
    T_0=5.0, a=-0.1, b=2, c=1.2, r_t=100.0, T_min=2.0, r_cool=10.0, a_cool=-0.2
)
T = profile(r)
#
_ = plt.semilogx(r, T, 'b-', label='Vikhlinin Temperature Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Temperature (T)')
_ = plt.legend()
plt.show()
