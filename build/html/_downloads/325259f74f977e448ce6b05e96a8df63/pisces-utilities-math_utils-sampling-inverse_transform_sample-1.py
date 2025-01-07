import numpy as np
import matplotlib.pyplot as plt
from pisces.utilities.math_utils.sampling import inverse_transform_sample
x = np.linspace(-5, 5, 1000)
pdf = np.exp(-0.5 * x**2)  # Gaussian PDF (unnormalized)
samples = inverse_transform_sample(x, pdf, 10000)
_ = plt.hist(samples, bins=50, density=True, alpha=0.6, label="Samples")
_ = plt.plot(x, pdf / np.trapz(pdf, x), label="PDF", color="red")
_ = plt.legend()
plt.show()
