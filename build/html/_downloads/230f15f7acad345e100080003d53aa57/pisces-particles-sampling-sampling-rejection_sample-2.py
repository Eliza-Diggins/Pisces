from scipy.stats import norm
import matplotlib.pyplot as plt
from pisces.particles.sampling.sampling import rejection_sample
import numpy as np
x_grid = np.linspace(-6,6,100)
y_grid = np.exp(-(x_grid**2)/2)
proposal = -np.abs(x_grid)/3 + 1
proposal = np.amax([y_grid,proposal],axis=0)
samples = rejection_sample(x_grid,y_grid,1_000_000,proposal=proposal,paxis=0)
hist, edges = np.histogram(samples, bins=100, density=True)
bin_centers = (edges[1:]+edges[:-1])/2
_ = plt.plot(bin_centers, hist)
_ = plt.plot(bin_centers, norm.pdf(bin_centers))
plt.show()
