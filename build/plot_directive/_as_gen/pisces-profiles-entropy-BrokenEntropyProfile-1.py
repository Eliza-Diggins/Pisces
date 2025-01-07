import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.entropy import BrokenEntropyProfile
#
r = np.logspace(1, 3, 100)
profile = BrokenEntropyProfile(r_s=300.0, K_scale=200.0, alpha=1.1, K_0=0.0)
K = profile(r)
#
_ = plt.loglog(r, K, 'b-', label='Broken Entropy Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Entropy (K)')
_ = plt.legend()
plt.show()
