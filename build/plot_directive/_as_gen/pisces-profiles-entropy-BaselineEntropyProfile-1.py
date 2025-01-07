import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.entropy import BaselineEntropyProfile
#
r = np.linspace(10, 1000, 100)
profile = BaselineEntropyProfile(K_0=10.0, K_200=200.0, r_200=1000.0, alpha=1.1)
K = profile(r)
#
_ = plt.loglog(r, K, 'r-', label='Baseline Entropy Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Entropy (K)')
_ = plt.legend()
plt.show()
