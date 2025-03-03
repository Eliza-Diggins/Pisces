from scipy.stats import norm,beta
import matplotlib.pyplot as plt
from pisces.particles.sampling.sampling import rejection_sample
import numpy as np
x,y = np.linspace(-5,5,500),np.linspace(0,1,500)
X,Y = np.meshgrid(x,y,indexing='ij')
C = np.moveaxis(np.asarray([X,Y]),0,-1)
px, py = lambda x: 0.5*(norm(loc=-3).pdf(x)+norm(loc=3).pdf(x)), beta(a=2,b=3).pdf
Z = px(X)*py(Y)
samples = rejection_sample(C,Z,100_000_000)
hist,ex,ey = np.histogram2d(*samples.T, bins=100, density=True)
ecx,ecy = (ex[1:]+ex[:-1])/2,(ey[1:]+ey[:-1])/2
Ix = np.sum(hist*np.diff(ey),axis=1)
Iy = np.sum(hist*np.diff(ex),axis=0)
fig,axes = plt.subplots(2,2,gridspec_kw=dict(hspace=0,wspace=0,height_ratios=[1,3],width_ratios=[3,1]))
_ = axes[0,1].set_visible(False)
_ = axes[1,0].imshow(hist.T,extent=(-5,5,0,1),origin='lower',aspect='auto')
_ = axes[0,0].plot(ecx,Ix,color='red',ls='-')
_ = axes[1,1].plot(Iy,ecy,color='blue',ls='-')
_ = axes[0,0].plot(ecx,px(ecx),color='red',ls=':')
_ = axes[1,1].plot(py(ecy),ecy,color='blue',ls=':')
plt.show()
