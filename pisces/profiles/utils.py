"""
Utility functions for profiles.
"""
from typing import Callable
import numpy as np
from scipy.integrate import quad
from utilities.math_utils.numeric import integrate_from_zero, integrate

def compute_ellipsoid_psi(function: Callable,
                          r_min: float,
                          r_max: float,
                          n_points: int = 1000,
                          boundary_mode: str = 'integrate',
                          boundary_behavior: int = (-1,-3),
                          spacing: str = 'log') -> tuple[np.ndarray, np.ndarray, float]:
    r"""
    Compute the ellipsoidal function :math:`\psi(m)` from quadrature.

    In the solution of the Poisson problem in ellipsoidal coordinate systems, the function

    .. math::

        \psi(m) = 2 \int_0^m \; d\xi \xi \rho(\xi)

    arises frequently. This function computes :math:`\psi(m)` at a set of points specified by ``r_min``, ``r_max``,
    and ``n_points``. It also computes the behavior of :math:`\psi(\infty)`.

    Parameters
    ----------
    function : Callable
        The density profile :math:`\rho(r)` as a function of radius. This must be a function of one variable with
        an asymptotic behavior that goes to zero faster than :math:`r^{-2}`.
    r_min : float
        The minimum radius for the computation. This should be in the native length units suitable for the problem at hand.
    r_max : float
        The maximum radius for the computation. This should be in the native length units suitable for the problem at hand.
    n_points : int, optional
        Number of points for the integration grid. Default is 1000. This is coupled with the ``spacing`` option to
        determine the distribution of the integration points.
    boundary_mode : str, optional
        Method for handling the boundary behavior, either ``'integrate'`` or ``'asymptotic'``. Default is ``'integrate'``. If ``'integrate'``,
        the callable provided in ``function`` is numerically integrated up to infinity. If ``'asymptotic'``, the ``boundary_behavior``
        is used to compute an asymptotic estimate.
    boundary_behavior : tuple of int, optional
        Specifies the asymptotic power-law behavior of the density profile at large and small radii.
        Required for ``'asymptotic'`` mode. The first value is the power-law behavior for small :math:`r` and the second is the
        behavior for large :math:`r`.
        The outer value must be a value smaller than :math:`-2`. Default is :math:`-3`.
        The inner value must be a value larger than :math:`-2`. Default is :math:`-1`.
    spacing : str, optional
        Spacing of the radius grid, either ``'log'`` or ``'linear'``. Default is ``'log'``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        A tuple containing:

        - Radii array :math:`r` (np.ndarray).
        - Computed :math:`\psi(m)` values for the given radii (np.ndarray).
        - Computed :math:`\psi(\infty)` (float), representing the contribution beyond ``r_max``.

    Notes
    -----
    - The function assumes that :math:`\rho(r)` is well-behaved (continuous and differentiable) within the range
      :math:`[r_{\text{min}}, r_{\text{max}}]` and beyond.
    - For ``'asymptotic'`` boundary handling, the density profile is approximated by a power law:
      :math:`\rho(r) \sim r^l` with :math:`l < -2`. Ensure the provided ``boundary_behavior`` matches the actual
      behavior of the density profile.

    Examples
    --------
    Let's compute :math:`\psi(r)` for the standard Hernquist density profile (:py:class:`~pisces.geometry.density.HernquistDensityProfile`).

    Because the Hernquist profile takes the form (with all constants set to unity)

    .. math::

        \rho(r) = \frac{1}{r(r+1)^3},

    We have that

    .. math::

        \psi(r) = 2\int_0^r \; d\xi \frac{1}{(\xi + 1)^3} = \frac{r(r+2)}{(r+1)^2}.

    Taking the limit,

    .. math::

        \lim_{r \to \infty} \psi(r) = 1.

    We therefore want to see this reproduced in our calculation.

    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import HernquistDensityProfile
        >>> from pisces.profiles.utils import compute_ellipsoid_psi

        First, we need to construct the density profile.

        >>> density_profile = HernquistDensityProfile(rho_0=1,r_s=1)

        Now we can solve for the function :math:`\psi(r)`.

        >>> rmin, rmax = 1e-2,1e2
        >>> rad, p, p_inf = compute_ellipsoid_psi(density_profile, rmin, rmax)
        >>> print(f'psi(infty) within machine precision = {np.isclose(p_inf,1,rtol=1e-7)}')
        psi(infty) within machine precision = True

        Let's make the theoretical function...

        >>> p_theory = lambda _r: (_r*(_r+2))/(_r+1)**2

        Now let's plot it

        >>> fig,axes = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[5,1]})
        >>> _ = axes[0].loglog(rad,p,'r-',label=r'$\psi(r)$ - Computed')
        >>> _ = axes[0].loglog(rad,p_theory(rad),'b:',label=r'$\psi(r)$ - Theoretical')
        >>> abs_err = np.abs((p-p_theory(rad))/p_theory(rad))
        >>> _ = axes[1].loglog(rad,abs_err)
        >>> _ = axes[0].set_ylabel(r"$\psi(r)$")
        >>> _ = axes[1].set_xlabel(r"$r$")
        >>> _ = axes[1].set_ylabel(r"abs. rel. err.")
        >>> _ = axes[0].legend()
        >>> plt.show()

    What happens if we don't have a callable form of the function? We can still utilize this method; however, the
    asymptotic behavior must now be specified and may be subject to additional errors. To make this work, we need
    to interpolate using a univariate spline (``scipy.interpolate.InterpolatedUnivariateSpline``).

    .. warning::

        If a spline is being used, it will not have the correct asymptotic behavior naturally. We therefore
        need to specify the boundary behavior manually.

    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import HernquistDensityProfile
        >>> from scipy.interpolate import InterpolatedUnivariateSpline
        >>> from pisces.profiles.utils import compute_ellipsoid_psi

        First, we need to construct the density profile.

        >>> dp = HernquistDensityProfile(rho_0=1,r_s=1)
        >>> rmin, rmax = 1e-2,1e2
        >>> r = np.geomspace(rmin,rmax,1000)
        >>> dens = dp(r)
        >>> density_spline = InterpolatedUnivariateSpline(r,dens)

        Now we can solve for the function :math:`\psi(r)`. Let's try with two different scenarios. First, we'll
        apply the case without fixing the boundary behavior, then we'll try it with a fixed boundary behavior.

        >>> rad_free, p_free, p_free_inf = compute_ellipsoid_psi(density_spline, rmin, rmax)
        >>> rad_fixed,p_fixed, p_fixed_inf = compute_ellipsoid_psi(density_spline, rmin, rmax, boundary_mode='asymptotic',boundary_behavior=(-1,-4))

        >>> p_theory = lambda _r: (_r*(_r+2))/(_r+1)**2

        Now let's plot it

        >>> fig,axes = plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[5,1]})
        >>> _ = axes[0].loglog(rad_free,p_free,'r-',label=r'$\psi(r)$ - Computed Free')
        >>> _ = axes[0].loglog(rad_fixed,p_fixed,'g-',label=r'$\psi(r)$ - Computed Fixed')
        >>> _ = axes[0].loglog(rad_free,p_theory(rad_free),'b:',label=r'$\psi(r)$ - Theoretical')
        >>> abs_err_free = np.abs((p_free-p_theory(rad_free))/p_theory(rad_free))
        >>> abs_err_fixed = np.abs((p_fixed-p_theory(rad_fixed))/p_theory(rad_fixed))
        >>> _ = axes[1].loglog(rad_free,abs_err_free,'r-',label=r'$\psi(r)$ - Computed Free')
        >>> _ = axes[1].loglog(rad_fixed,abs_err_fixed,'g-',label=r'$\psi(r)$ - Computed Fixed')
        >>> _ = axes[0].set_ylabel(r"$\psi(r)$")
        >>> _ = axes[1].set_xlabel(r"$r$")
        >>> _ = axes[1].set_ylabel(r"abs. rel. err.")
        >>> _ = axes[0].legend()
        >>> plt.show()

    .. note::

        Clearly, its optimal to have the full function known to arbitrary precision; however, in cases where that
        is not possible, the spline approach will produce reasonable accurate solutions.

    Warnings
    --------
    - If the density profile does not asymptotically decay faster than :math:`r^{-2}`, the integral for
      :math:`\psi(m)` will diverge.
    - The ``boundary_behavior`` parameter must reflect the actual power-law behavior of the density profile
      when ``boundary_mode='asymptotic'``.

    See Also
    --------
    :py:func:`~pisces.utilities.math.integrate_from_zero`: Computes definite integrals from 0 to a specified upper bound.
    :py:func:`scipy.integrate.quad`: Performs numerical integration.

    Raises
    ------
    ValueError
        If ``spacing`` is not ``'log'`` or ``'linear'``.
    ValueError
        If ``boundary_mode`` is not ``'integrate'`` or ``'asymptotic'``.
    ValueError
        If ``boundary_behavior >= -2`` in ``'asymptotic'`` mode.
    """
    # Construct the radii array using r_min, r_max, and n_points.
    if spacing == 'log':
        radii = np.geomspace(r_min, r_max, num=n_points)
    elif spacing == 'linear':
        radii = np.linspace(r_min, r_max, num=n_points)
    else:
        raise ValueError('spacing must be "log" or "linear"')

    # Generate the integrand for the principal integration.
    integrand = lambda _r: _r * function(_r)

    # Compute psi(infty) based on the boundary mode.
    if boundary_mode == 'integrate':
        # Perform numerical integration to infinity.
        psi = 2 * integrate_from_zero(integrand, radii)
        psi_infty = psi[-1] + (2 * quad(integrand, r_max, np.inf)[0])
    elif boundary_mode == 'asymptotic':
        # Asymptotic approximation: assume :math:`\rho(r) \sim r^l` with :math:`l < -2`.
        if boundary_behavior[1] >= -2:
            raise ValueError("For asymptotic behavior, outer boundary_behavior (power index) must be < -2.")
        if boundary_behavior[0] <= -2:
            raise ValueError("For asymptotic behavior, inner boundary_behavior (power index) must be > -2.")
        # Compute raw psi
        psi = 2 * integrate(integrand,radii,x_0=radii[0],minima=True)
        rho_at_rmin = function(r_min)
        psi_core = (2 * rho_at_rmin * r_min**2 / (boundary_behavior[0] + 2))
        psi += psi_core

        rho_at_rmax = function(r_max)
        psi_infty = psi[-1] + (-2 * rho_at_rmax * r_max**2 / (boundary_behavior[1] + 2))
    else:
        raise ValueError('boundary_mode must be "integrate" or "asymptotic"')

    return radii, psi, psi_infty