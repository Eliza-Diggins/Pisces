from pisces.profiles.density import HernquistDensityProfile
import matplotlib.pyplot as plt
density_profile = HernquistDensityProfile(rho_0=5,r_s=1)
_inner_lim = density_profile.get_limiting_behavior(limit='inner')
_outer_lim = density_profile.get_limiting_behavior(limit='outer')
#
# Let's now use this information to make a plot of our profile and its limiting behavior:
#
fig,axes = plt.subplots(1,1)
r = np.logspace(-3,3,1000)
_ = axes.loglog(r,density_profile(r),label=r'$\rho(r)$')
_ = axes.loglog(r, _outer_lim[0]*(r**_outer_lim[1]), label=r'$\lim_{r \to \infty} \rho(r)$')
_ = axes.loglog(r, _inner_lim[0]*(r**_inner_lim[1]), label=r'$\lim_{r \to 0} \rho(r)$')
_ = axes.set_ylim([1e-6,1e5])
_ = axes.set_xlim([1e-3,1e3])
_ = axes.set_xlabel('r')
_ = axes.set_ylabel('density(r)')
_ = axes.legend(loc='best')
plt.show()
