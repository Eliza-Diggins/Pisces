import numpy as np
import matplotlib.pyplot as plt
from pisces.utilities.math_utils.sampling import rejection_sample
xmin,xmax = -2,2
x,y = np.linspace(xmin,xmax,1000),np.linspace(xmin,xmax,1000)
x_hist,y_hist = np.linspace(xmin,xmax,100),np.linspace(xmin,xmax,100)
X,Y = np.meshgrid(x,y,indexing='ij') # The indexing is CRITICAL here.
abscissa = np.moveaxis(np.stack([X,Y],axis=0),0,-1)
field = np.exp(-1*(2*abscissa[...,0]**2 + abscissa[...,1]**2))
samples = rejection_sample(abscissa,field,1000000,chunk_size=10000)
hist_image,_,_ = np.histogram2d(samples[:,0],samples[:,1],bins=(x_hist,y_hist))
fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,5),gridspec_kw={'wspace':0.1})
_ = axes[0].imshow(hist_image.T/np.amax(hist_image.T), origin='lower', extent=[xmin,xmax,xmin,xmax],cmap='inferno')
_ = axes[1].imshow(field.T/np.amax(field.T),origin='lower',extent=[xmin,xmax, xmin,xmax],cmap='inferno')
_ = plt.colorbar(plt.cm.ScalarMappable(cmap='inferno'),ax=axes,fraction=0.07)
_ = axes[0].set_ylabel('y')
_ = axes[1].set_xlabel('x')
_ = axes[0].set_xlabel('x')
_ = axes[1].set_title("Likelihood Function")
_ = axes[0].set_title("Sampled Values")
plt.show()
