from pisces.particles.sampling import sample_inverse_cumulative
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
x_grid = np.linspace(-5,5,100)
y_grid = np.exp(-(x_grid**2)/2)
samples = sample_inverse_cumulative(x_grid,y_grid,1_000_000,bounds=(-6,6))
hist, edges = np.histogram(samples, bins=100, density=True)
bin_centers = (edges[1:]+edges[:-1])/2
_ = plt.plot(bin_centers, hist)
_ = plt.plot(bin_centers, norm.pdf(bin_centers))
plt.show()
