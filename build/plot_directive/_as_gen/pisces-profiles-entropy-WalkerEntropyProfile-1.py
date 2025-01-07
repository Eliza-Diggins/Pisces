import numpy as np
import matplotlib.pyplot as plt
from pisces.profiles.entropy import WalkerEntropyProfile
#
r = np.linspace(0, 2000, 200)
profile = WalkerEntropyProfile(r_200=1000.0, A=0.5, B=0.2, K_scale=100.0, alpha=1.1)
K = profile(r)
#
_ = plt.semilogy(r, K, 'g-', label='Walker Entropy Profile')
_ = plt.xlabel('Radius (r)')
_ = plt.ylabel('Entropy (K)')
_ = plt.legend()
plt.show()
