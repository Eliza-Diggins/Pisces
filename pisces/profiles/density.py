"""
Density profiles for astrophysical modeling.
"""
from abc import ABC
from typing import List, Tuple, Dict, Any
import sympy as sp
import numpy as np
from pisces.profiles.base import RadialProfile


class RadialDensityProfile(RadialProfile,ABC):
    r"""
    Base class for radial density profiles with fixed axes, units, and parameters.
    """
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES =  ['r']
    PARAMETERS = None
    UNITS: str = "Msun/kpc"

    def get_ellipsoid_psi(self,r_min: float,r_max: float, n_points: int, scale: str = 'log'):
        r"""
        Compute the density moment function for use in computing the ellipsoidal potential.

        For ellipsoidally distributed systems, the gravitational potential relies on the function

        .. math::

            \psi(r) = \int_0^r \; d\xi \; \xi \rho(\xi)

        in quadrature. This method provides the values of this function over a range of :math:`r` values, including
        at :math:`r=0`, and :math:`r=\infty`.

        .. note::

            It is not necessarily the case that all density profiles have a well defined :math:`\psi` function. For those
            which do not, an error is raised in this method.

        Parameters
        ----------
        r_min : float
            The minimum radius for the radial range of the computation. Must be greater than zero.

            .. note::

                The returned abscissa values will be a length ``n_points+1`` array, with the first element corresponding
                to :math:`\Psi(0)`.

        r_max : float
            The maximum radius for the radial range of the computation.
        n_points : int
            The number of points to use in the radial grid for interpolation.
        scale : str, optional
            The scale of the radial grid. Options are:
            - ``'log'``: Geometric spacing for radii.
            - ``'linear'``: Linear spacing for radii.
            Default is ``'log'``.

        Returns
        -------
        radii : numpy.ndarray
            The radial grid including zero, with size ``n_points + 1``.
        psi_integral : numpy.ndarray
            The values of the :math:`\psi(r)` function over the range specified by ``radii``. This includes the zero point
            value.
        psi_inf : float
            The asymptotic value of the potential as :math:`r \\to \\infty`. This is generally necessary independently
            in the computation.

        Raises
        ------
        ValueError
            If:

            - The behavior of :math:`r^2 \rho(r)` as :math:`r \to 0` does not converge.
            - ``r_min`` is negative.
            - An unknown value is provided for ``scale``.

        Notes
        -----

        **Convergence Check:**

        The method first validates that the integral of :math:`r^2 \rho(r)` converges as
        :math:`r \to 0`. If it diverges, the ellipsoidal potential function is undefined
        because the value at :math:`r = 0` cannot be established.

        The zero-point value is set to zero under the assumption that the integral
        from :math:`0` to :math:`r_{min}` converges to zero.

        **Integration Details:**

        The method constructs the radial grid using either linear or logarithmic spacing
        (based on the ``scale`` parameter) and numerically integrates the density profile
        from :math:`0` to :math:`r`. The quadrature is extended to infinity to determine the
        asymptotic potential value.

        Examples
        --------
        The NFW profile follows the form

        .. math::

            \rho(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)\left(1+\frac{r}{r_s}\right)^2},

        Which has an analytic form for :math:`\psi(r)`:

        .. math::

            \psi(r) = \frac{\rho_0 r_s r}{r_s + r}.

        Let's test this prediction against our numerical solution:

        .. plot::

            >>> from pisces.profiles.density import NFWDensityProfile
            >>> import matplotlib.pyplot as plt
            >>> profile = NFWDensityProfile(rho_0=1.0, r_s=1.0)
            >>> r_min, r_max, n_points = 1e-2, 1000, 1000
            >>> radii, psi_integral, psi_inf = profile.get_ellipsoid_psi(r_min, r_max, n_points, scale='log')

            >>> fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw=dict(height_ratios=[4,1],hspace=0))
            >>> theory = radii/(1+radii)
            >>> _ = axes[0].plot(radii, psi_integral,color='black', label=r'$\psi(r)$ - Computed')
            >>> _ = axes[0].plot(radii, theory,color='cyan',ls='--', label=r'$\psi(r)$ - Theory')
            >>> _ = axes[0].axhline(psi_inf, color='red', linestyle='--', label=r'$\lim_{r\to \infty} \psi(r)$')
            >>> _ = axes[0].axhline(psi_integral[0], color='red', linestyle=':', label=r'$\lim_{r \to 0} \psi(r)$')
            >>> _ = axes[0].set_xscale('log')
            >>> _ = axes[0].set_ylabel(r'$\psi(r)$')
            >>> _ = axes[1].set_xlabel('Radius (r)')
            >>> _ = axes[1].plot(radii[1:], np.abs((psi_integral-theory)/theory)[1:], color='black', label=r'Error')
            >>> _ = axes[1].set_xscale('log')
            >>> _ = axes[1].set_yscale('log')
            >>> _ = plt.legend()
            >>> plt.show()

        """
        # Validate that the integral actually converges for this system.
        # This requires us checking the boundary behavior as r -> 0 and ensuring that it goes faster
        # than r**-2. If it does not, we cannot establish the psi(0) value which will lead to issues
        # in the quadrature.
        from pisces.utilities.math_utils.numeric import integrate_from_zero
        from scipy.integrate import quad
        rad_symbol = self.SYMBAXES[0]
        limit = float(sp.limit(self.symbolic_expression*(rad_symbol**2),rad_symbol, 0, "-"))

        if np.isinf(limit) or np.isnan(limit):
            raise ValueError(f"The behavior of r^2 * rho(r) as r -> 0 does not converge: {limit}.\n"
                             f"The ellipsoidal psi function therefore has no value at zero and is not valid "
                             f"for use in this context.")

        # If the limit exists, it must be zero (int_0^0 f(x) dx = 0), so we can immediately set the
        # zero point value.
        zero_point_value = 0

        # Construct the abscissa and prepare for the interpolation step of this scheme. We interpolate
        # including zero (using the zero_point_value).
        if r_min < 0:
            raise ValueError("The minimum radius may not be negative.")

        # Construct the abscissa.
        if scale == 'log':
            radii = np.geomspace(r_min,r_max,n_points)
        elif scale == 'linear':
            radii = np.linspace(r_min,r_max,n_points)
        else:
            raise ValueError(f"Unknown scale '{scale}'")

        # Perform the baseline integration over the standard (non-limit including) abscissa.
        integrand = lambda _xi: _xi*self(_xi)
        psi_integral = integrate_from_zero(integrand, radii)

        # add the zero-point to the abcissa and to the value set.
        psi_integral = np.concatenate([np.array([zero_point_value]), psi_integral])
        radii = np.concatenate([np.array([0]),radii])

        # Determine the limit at infinity. This can be achieved by integrating in quadrature
        # because our density profile is fully characterized over the entire real line.
        psi_inf = psi_integral[-1]+ quad(integrand, float(radii[-1]), np.inf)[0]

        return radii, psi_integral, psi_inf


class NFWDensityProfile(RadialDensityProfile):
    r"""
    Navarro-Frenk-White (NFW) Density Profile.

    This profile is commonly used in astrophysics to describe the dark matter halo density
    in a spherical, isotropic system. It is derived from simulations of structure formation.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.


    References
    ----------
    .. [NaFrWh96] Navarro, Frenk, and White, 1996.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import NFWDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = NFWDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho,'k-', label='NFW Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, CoredNFWDensityProfile, SingularIsothermalDensityProfile
    """

    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

class HernquistDensityProfile(RadialDensityProfile):
    r"""
    Hernquist Density Profile.

    This profile is often used to model the density distribution of elliptical galaxies and bulges.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^3}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Her90] Hernquist, L. 1990.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import HernquistDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = HernquistDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho,'k-', label='Hernquist Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, SingularIsothermalDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)

class EinastoDensityProfile(RadialDensityProfile):
    r"""
    Einasto Density Profile.

    This profile provides a flexible model for dark matter halos with a gradual density decline.

    .. math::
        \rho(r) = \rho_0 \exp\left(-2 \alpha \left[\left(\frac{r}{r_s}\right)^\alpha - 1\right]\right)

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`\alpha` is a shape parameter that controls the profile steepness.

    References
    ----------
    .. [Ein65] Einasto, J., 1965.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import EinastoDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = EinastoDensityProfile(rho_0=1.0, r_s=1.0, alpha=0.18)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Einasto Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, CoredNFWDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
        "alpha": 0.18,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0, alpha=0.18):
        return rho_0 * sp.exp(-2 * alpha * ((r / r_s) ** alpha - 1))

class SingularIsothermalDensityProfile(RadialDensityProfile):
    r"""
    Singular Isothermal Sphere (SIS) Density Profile.

    The SIS profile is a simple model commonly used to describe the density distribution
    of dark matter in galaxies and galaxy clusters under the assumption of an isothermal system.

    .. math::
        \rho(r) = \frac{\\rho_0}{r^2}

    where:

    - :math:`\rho_0` is the central density.

    References
    ----------
    .. [BinTr87] Binney, J. & Tremaine, S., 1987.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import SingularIsothermalDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = SingularIsothermalDensityProfile(rho_0=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='SIS Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, CoredIsothermalDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0):
        return rho_0 / r**2

class CoredIsothermalDensityProfile(RadialDensityProfile):
    r"""
    Cored Isothermal Sphere Density Profile.

    This profile modifies the Singular Isothermal Sphere (SIS) by introducing a core radius
    to account for the central flattening of the density distribution.

    .. math::
        \rho(r) = \frac{\\rho_0}{1 + \left(\frac{r}{r_c}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.

    References
    ----------
    .. [Bur95] Burkert, A., 1995.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import CoredIsothermalDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = CoredIsothermalDensityProfile(rho_0=1.0, r_c=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Cored Isothermal Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    SingularIsothermalDensityProfile, NFWDensityProfile, BurkertDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_c": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

class PlummerDensityProfile(RadialDensityProfile):
    r"""
    Plummer Density Profile.

    The Plummer profile is commonly used to model the density distribution of star clusters
    or spherical galaxies. It features a central core and a steep falloff at larger radii.

    .. math::
        \rho(r) = \frac{3M}{4\pi r_s^3} \left(1 + \left(\frac{r}{r_s}\right)^2\right)^{-5/2}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Plu11] Plummer, H. C., 1911.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import PlummerDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = PlummerDensityProfile(M=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Plummer Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, NFWDensityProfile, JaffeDensityProfile
    """
    AXES = ['r']
    
    PARAMETERS = {
        "M": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, M=1.0, r_s=1.0):
        return (3 * M) / (4 * sp.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

class DehnenDensityProfile(RadialDensityProfile):
    r"""
    Dehnen Density Profile.

    This profile is widely used in modeling galactic bulges and elliptical galaxies.
    It generalizes other profiles like Hernquist and Jaffe with an adjustable inner slope.

    .. math::
        \rho(r) = \frac{(3 - \gamma)M}{4\pi r_s^3}
        \left(\frac{r}{r_s}\right)^{-\gamma} \left(1 + \frac{r}{r_s}\right)^{\gamma - 4}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.
    - :math:`\gamma` controls the inner density slope.

    References
    ----------
    .. [Deh93] Dehnen, W., 1993.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import DehnenDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = DehnenDensityProfile(M=1.0, r_s=1.0, gamma=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Dehnen Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, JaffeDensityProfile, NFWDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "M": 1.0,
        "r_s": 1.0,
        "gamma": 1.0,
    }

    @staticmethod
    def _function(r, M=1.0, r_s=1.0, gamma=1.0):
        return (
            ((3 - gamma) * M)
            / (4 * sp.pi * r_s**3)
            * (r / r_s) ** (-gamma)
            * (1 + r / r_s) ** (gamma - 4)
        )

class JaffeDensityProfile(RadialDensityProfile):
    r"""
    Jaffe Density Profile.

    This profile is commonly used to describe the density distribution of elliptical galaxies.

    .. math::
        \rho(r) = \frac{\\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Jaf83] Jaffe, W., 1983.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import JaffeDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = JaffeDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Jaffe Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, DehnenDensityProfile, NFWDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)


class KingDensityProfile(RadialDensityProfile):
    r"""
    King Density Profile.

    This profile describes the density distribution in globular clusters and galaxy clusters,
    accounting for truncation at larger radii.

    .. math::
        \rho(r) = \\rho_0 \\left[\left(1 + \left(\frac{r}{r_c}\right)^2\\right)^{-3/2}
        - \left(1 + \left(\frac{r_t}{r_c}\right)^2\\right)^{-3/2}\\right]

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_t` is the truncation radius.

    References
    ----------
    .. [Kin66] King, I. R., 1966.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import KingDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = KingDensityProfile(rho_0=1.0, r_c=1.0, r_t=5.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='King Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, BurkertDensityProfile, PlummerDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_c": 1.0,
        "r_t": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_c=1.0, r_t=1.0):
        return rho_0 * ((1 + (r / r_c) ** 2) ** (-3 / 2) - (1 + (r_t / r_c) ** 2) ** (-3 / 2))


class BurkertDensityProfile(RadialDensityProfile):
    r"""
    Burkert Density Profile.

    This profile describes dark matter halos with a flat density core, often used to
    fit rotation curves of dwarf galaxies.

    .. math::
        \rho(r) = \frac{\\rho_0}{\left(1 + \frac{r}{r_s}\right) \left(1 + \left(\frac{r}{r_s}\right)^2\right)}

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Bur95] Burkert, A., 1995.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import BurkertDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = BurkertDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Burkert Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, KingDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))


class MooreDensityProfile(RadialDensityProfile):
    r"""
    Moore Density Profile.

    This profile describes the density of dark matter halos with a steeper central slope compared to NFW.

    .. math::
        \rho(r) = \frac{\\rho_0}{\left(\frac{r}{r_s}\\right)^{3/2} \left(1 + \frac{r}{r_s}\\right)^{3/2}}

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Moo98] Moore, B., et al., 1998.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import MooreDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = MooreDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Moore Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, HernquistDensityProfile
    """
    AXES = ['r']
    
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))


class CoredNFWDensityProfile(RadialDensityProfile):
    r"""
    Cored Navarro-Frenk-White (NFW) Density Profile.

    This profile modifies the standard NFW profile by introducing a core, leading to
    a shallower density slope near the center.

    .. math::
        \rho(r) = \frac{\\rho_0}{\left(1 + \left(\frac{r}{r_s}\right)^2\right) \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    References
    ----------
    .. [Ric05] Ricotti, M., et al., 2005.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import CoredNFWDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = CoredNFWDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Cored NFW Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, BurkertDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)


class VikhlininDensityProfile(RadialDensityProfile):
    r"""
    Vikhlinin Density Profile.

    This profile is used to model the density of galaxy clusters, incorporating
    a truncation at large radii and additional flexibility for inner slopes.

    .. math::
        \rho(r) = \rho_0 \left(\frac{r}{r_c}\right)^{-0.5 \alpha}
        \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-1.5 \beta + 0.25 \alpha}
        \left(1 + \left(\frac{r}{r_s}\right)^{\gamma}\right)^{-0.5 \epsilon / \gamma}

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_s` is the truncation radius.
    - :math:`\alpha, \beta, \gamma, \epsilon` control the slope and truncation behavior.

    References
    ----------
    .. [Vik06] Vikhlinin, A., et al., 2006.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import VikhlininDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = VikhlininDensityProfile(rho_0=1.0, r_c=1.0, r_s=5.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='Vikhlinin Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    KingDensityProfile, NFWDensityProfile, BurkertDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_c": 1.0,
        "r_s": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "epsilon": 1.0,
        "gamma": 3.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )


class AM06DensityProfile(RadialDensityProfile):
    r"""
    An & Zhao (2006) Density Profile (AM06).

    This density profile is a generalized model that allows flexibility in fitting
    the density distributions of dark matter halos. It includes additional parameters
    for controlling inner and outer slopes, truncation, and other scaling properties.

    .. math::
        \rho(r) = \rho_0 \left(1 + \frac{r}{a_c}\right) \left(1 + \frac{r}{a_c c}\right)^{\alpha} \left(1 + \frac{r}{a}\right)^{\beta}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`a_c` is the core radius.
    - :math:`c` is the concentration parameter.
    - :math:`a` is the scale radius.
    - :math:`\alpha` controls the slope of the transition near the core.
    - :math:`\beta` controls the outer slope.

    Use Case
    --------
    This profile is well-suited for modeling dark matter halos with detailed inner and outer slope behaviors.

    References
    ----------
    .. [An06] An, J., & Zhao, H. S. (2006). Mon. Not. R. Astron. Soc.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import AM06DensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = AM06DensityProfile(rho_0=1.0, a_c=1.0, c=2.0, a=3.0, alpha=1.0, beta=3.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='AM06 Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, VikhlininDensityProfile, HernquistDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "a": 1.0,
        "a_c": 1.0,
        "c": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, a=1.0, a_c=1.0, c=1.0, alpha=1.0, beta=1.0):
        return rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta


class SNFWDensityProfile(RadialDensityProfile):
    r"""
    Simplified Navarro-Frenk-White (SNFW) Density Profile.

    This profile is a simplified version of the NFW profile, widely used for modeling
    dark matter halos with specific scaling.

    .. math::
        \rho(r) = \frac{3M}{16\pi a^3} \frac{1}{\frac{r}{a} \left(1 + \frac{r}{a}\right)^{2.5}}

    where:

    - :math:`M` is the total mass.
    - :math:`a` is the scale radius.

    References
    ----------
    .. [Zha96] Zhao, H., 1996.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import SNFWDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = SNFWDensityProfile(M=1.0, a=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='SNFW Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, TNFWDensityProfile, HernquistDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "M": 1.0,
        "a": 1.0,
    }

    @staticmethod
    def _function(r, M=1.0, a=1.0):
        return 3.0 * M / (16.0 * sp.pi * a**3) / ((r / a) * (1.0 + r / a) ** 2.5)


class TNFWDensityProfile(RadialDensityProfile):
    r"""
    Truncated Navarro-Frenk-White (TNFW) Density Profile.

    This profile is a modification of the NFW profile with an additional truncation
    term to account for finite halo sizes.

    .. math::
        \rho(r) = \frac{\\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}
        \frac{1}{1 + \left(\frac{r}{r_t}\right)^2}

    where:

    - :math:`\\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`r_t` is the truncation radius.

    References
    ----------
    .. [Hay07] Hayashi, E., et al., 2007.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import TNFWDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = TNFWDensityProfile(rho_0=1.0, r_s=1.0, r_t=10.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, 'k-', label='TNFW Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, SNFWDensityProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "rho_0": 1.0,
        "r_s": 1.0,
        "r_t": 1.0,
    }

    @staticmethod
    def _function(r, rho_0=1.0, r_s=1.0, r_t=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2) / (1 + (r / r_t) ** 2)

if __name__ == '__main__':
    q = DehnenDensityProfile()
    print(q._get_ellipsoid_psi_spline(1e-6,1e6,100,scale='log'))