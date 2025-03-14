"""
Density profiles for use in constructing astrophysical models.
"""
from abc import ABC
from typing import Any, Dict

import sympy as sp
from unyt import unyt_quantity

from pisces.profiles.base import CylindricalProfile, RadialProfile, class_expression


class _RadialDensityProfile(RadialProfile, ABC):
    r"""
    Base class for radial density profiles. This class provides a location in which to define
    various derived quantities which are shared across all the radial density profiles.
    """
    _is_parent_profile = True


class NFWDensityProfile(_RadialDensityProfile):
    r"""
    Navarro-Frenk-White :footcite:p:`NFWProfile` (NFW) Density Profile.

    This profile is commonly used in astrophysics to describe the dark matter halo density
    in a spherical, isotropic system. It is derived from simulations of structure formation.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`NFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius


    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`NFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 4\pi\,\rho_0\,r_s^3\,\bigl[\ln(1 + \tfrac{r}{r_s}) \;-\; \tfrac{r}{r + r_s}\bigr]`
             - None
           * - ``spherical_potential``
             - :math:`\Phi(r) = -\,\frac{4\pi\,\rho_0\,r_s^3}{r}\,\ln\!\Bigl(1 + \tfrac{r}{r_s}\Bigr)`
             - None
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) \;=\; 2\int_{0}^{r}\! \xi\,\rho(\xi)\,d\xi`
             - Inherited from base. Set to `on_demand=True`

    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

    @class_expression(name="spherical_potential", on_demand=False)
    @staticmethod
    def _spherical_potential(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, rho_0 = parameters["r_s"], parameters["rho_0"]

        # Produce the potential
        return -(4 * sp.pi * rho_0 * r_s**3) * sp.log(1 + (r / r_s)) / r

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, rho_0 = parameters["r_s"], parameters["rho_0"]

        # Produce the mass
        return (4 * sp.pi * rho_0 * r_s**3) * (
            sp.log(1 + (r / r_s)) - (r / (r_s + r))
        )


class HernquistDensityProfile(_RadialDensityProfile):
    r"""
    Hernquist Density Profile :footcite:p:`HernquistProfile`.

    This profile is often used to model the density distribution of elliptical galaxies and bulges.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^3}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Hernquist Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`HernquistDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 2\pi\,\rho_0\,r_s^3 \,\bigl(\tfrac{r}{r_s + r}\bigr)^{2}`
             - None
           * - ``spherical_potential``
             - :math:`\Phi(r) = -\,\frac{2\pi\,\rho_0\,r_s^3}{r + r_s}`
             - None
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)

    @class_expression(name="spherical_potential", on_demand=False)
    @staticmethod
    def _spherical_potential(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, rho_0 = parameters["r_s"], parameters["rho_0"]

        # Produce the potential
        return -(2 * sp.pi * rho_0 * r_s**3) / (r + r_s)

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, rho_0 = parameters["r_s"], parameters["rho_0"]

        # Produce the mass
        return (2 * sp.pi * rho_0 * r_s**3) * (r / (r_s + r)) ** 2


class EinastoDensityProfile(_RadialDensityProfile):
    r"""
    Einasto Density Profile :footcite:p:`EinastoProfile`.

    This profile provides a flexible model for dark matter halos with a gradual density decline.

    .. math::
        \rho(r) = \rho_0 \exp\left(-2 \alpha \left[\left(\frac{r}{r_s}\right)^\alpha - 1\right]\right)

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`\alpha` is a shape parameter that controls the profile steepness.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`EinastoDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Shape parameter controlling profile steepness



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`EinastoDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
        "alpha": unyt_quantity(0.18, ""),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0, alpha=0.18):
        return rho_0 * sp.exp(-2 * alpha * ((r / r_s) ** alpha - 1))


class SingularIsothermalDensityProfile(_RadialDensityProfile):
    r"""
    Singular Isothermal Sphere (SIS) Density Profile :footcite:p:`BinneyTremaine`.

    The SIS profile is a simple model commonly used to describe the density distribution
    of dark matter in galaxies and galaxy clusters under the assumption of an isothermal system.

    .. math::
        \rho(r) = \frac{\rho_0}{r^2}

    where:

    - :math:`\rho_0` is the central density.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`SingularIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`SingularIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 4\pi\,\rho_0\,r`
             - Diverges as :math:`r\to\infty`.
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0):
        return rho_0 / r**2

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        rho_0 = parameters["rho_0"]

        # Produce the mass
        return 4 * sp.pi * rho_0 * r


class CoredIsothermalDensityProfile(_RadialDensityProfile):
    r"""
    Cored Isothermal Sphere Density Profile :footcite:p:`BinneyTremaine`.

    This profile modifies the Singular Isothermal Sphere (SIS) by introducing a core radius
    to account for the central flattening of the density distribution.

    .. math::
        \rho(r) = \frac{\rho_0}{1 + \left(\frac{r}{r_c}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.

    .. dropdown:: Parameters

        .. list-table:: Cored Isothermal Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`CoredIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_c": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)


class PlummerDensityProfile(_RadialDensityProfile):
    r"""
    Plummer Density Profile :footcite:p:`PlummerProfile`.

    The Plummer profile is commonly used to model the density distribution of star clusters
    or spherical galaxies. It features a central core and a steep falloff at larger radii.

    .. math::
        \rho(r) = \frac{3M}{4\pi r_s^3} \left(1 + \left(\frac{r}{r_s}\right)^2\right)^{-5/2}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`PlummerDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`PlummerDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = M\,\bigl(\tfrac{r}{\sqrt{r^2 + r_s^2}}\bigr)^{3}`
             - None
           * - ``spherical_potential``
             - :math:`\Phi(r) = -\,\frac{M}{\sqrt{r^2 + r_s^2}}`
             - Multiplicative constants (e.g. `G`) can be included externally
           * - ``surface_density``
             - :math:`\Sigma(R) = \frac{M\,r_s^2}{\pi\,\bigl(r_s^2 + R^2\bigr)^{2}}`
             - On-demand expression for projected (2D) density
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    AXES = ["r"]

    DEFAULT_PARAMETERS = {
        "M": unyt_quantity(1.0, "Msun"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, M=1.0, r_s=1.0):
        return (3 * M) / (4 * sp.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

    @class_expression(name="spherical_potential", on_demand=False)
    @staticmethod
    def _spherical_potential(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, M = parameters["r_s"], parameters["M"]

        # Produce the potential
        return -M / sp.sqrt(r**2 + r_s**2)

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, M = parameters["r_s"], parameters["M"]

        # Produce the mass
        return M * (r / sp.sqrt(r**2 + r_s**2)) ** 3

    @class_expression(name="surface_density", on_demand=True)
    @staticmethod
    def _surface_density(axes, parameters, _):
        r = axes[0]
        r_s, M = parameters["r_s"], parameters["M"]
        return (M * r_s**2) / (sp.pi * (r_s**2 + r**2) ** 2)


class DehnenDensityProfile(_RadialDensityProfile):
    r"""
    Dehnen Density Profile :footcite:p:`DehnenProfile`.

    This profile is widely used in modeling galactic bulges and elliptical galaxies.
    It generalizes other profiles like Hernquist and Jaffe with an adjustable inner slope.

    .. math::
        \rho(r) = \frac{(3 - \gamma)M}{4\pi r_s^3}
        \left(\frac{r}{r_s}\right)^{-\gamma} \left(1 + \frac{r}{r_s}\right)^{\gamma - 4}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.
    - :math:`\gamma` controls the inner density slope.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`DehnenDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``gamma``
             - :math:`\gamma`
             - Controls the inner density slope



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`DehnenDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = M\,\Bigl(\frac{r}{r + r_s}\Bigr)^{3 - \gamma}`
             - Contains Hernquist (gamma=1) and Jaffe (gamma=2) as special cases
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "M": unyt_quantity(1.0, "Msun"),
        "r_s": unyt_quantity(1.0, "pc"),
        "gamma": unyt_quantity(1.0, ""),
    }

    @staticmethod
    def _profile(r, M=1.0, r_s=1.0, gamma=1.0):
        return (
            ((3 - gamma) * M)
            / (4 * sp.pi * r_s**3)
            * (r / r_s) ** (-gamma)
            * (1 + r / r_s) ** (gamma - 4)
        )

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, M, gamma = parameters["r_s"], parameters["M"], parameters["gamma"]

        # Produce the mass
        return M * (r / (r + r_s)) ** (3 - gamma)


class JaffeDensityProfile(_RadialDensityProfile):
    r"""
    Jaffe Density Profile :footcite:p:`JaffeProfile`.

    This profile is commonly used to describe the density distribution of elliptical galaxies.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`JaffeDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`JaffeDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 4\pi\,\rho_0\,r_s^3\,\Bigl[\ln(r + r_s) + \ln(r_s) \;-\; 1 \;+\; \frac{r_s}{r + r_s}\Bigr]`
             - (As coded; can simplify further)
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)

    @class_expression(name="spherical_mass", on_demand=False)
    @staticmethod
    def _spherical_mass(axes, parameters, _):
        # Grab the symbols out.
        r = axes[0]
        r_s, rho_0 = parameters["r_s"], parameters["rho_0"]

        # Produce the mass
        return (4 * sp.pi * r_s**3 * rho_0) * (
            (r_s / (r + r_s)) + sp.log(r_s + r) - 1 + sp.log(r_s)
        )


class KingDensityProfile(_RadialDensityProfile):
    r"""
    King Density Profile :footcite:p:`KingProfile`.

    This profile describes the density distribution in globular clusters and galaxy clusters,
    accounting for truncation at larger radii.

    .. math::
        \rho(r) = \rho_0 \left[\left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-3/2}
        - \left(1 + \left(\frac{r_t}{r_c}\right)^2\right)^{-3/2}\right]

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_t` is the truncation radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`KingDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``r_t``
             - :math:`r_t`
             - Truncation radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`KingDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_c": unyt_quantity(1.0, "pc"),
        "r_t": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_c=1.0, r_t=1.0):
        return rho_0 * (
            (1 + (r / r_c) ** 2) ** (-3 / 2) - (1 + (r_t / r_c) ** 2) ** (-3 / 2)
        )


class BurkertDensityProfile(_RadialDensityProfile):
    r"""
    Burkert Density Profile :footcite:p:`BurkertProfile`.

    This profile describes dark matter halos with a flat density core, often used to
    fit rotation curves of dwarf galaxies.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(1 + \frac{r}{r_s}\right) \left(1 + \left(\frac{r}{r_s}\right)^2\right)}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`BurkertDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`BurkertDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))


class MooreDensityProfile(_RadialDensityProfile):
    r"""
    Moore Density Profile :footcite:p:`MooreProfile`.

    This profile describes the density of dark matter halos with a steeper central slope compared to NFW.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)^{3/2} \left(1 + \frac{r}{r_s}\right)^{3/2}}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`MooreDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`MooreDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    AXES = ["r"]

    DEFAULT_PARAMETERS = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))


class CoredNFWDensityProfile(_RadialDensityProfile):
    r"""
    Cored Navarro-Frenk-White (NFW) Density Profile :footcite:p:`CNFWProfile`.

    This profile modifies the standard NFW profile by introducing a core, leading to
    a shallower density slope near the center.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(1 + \left(\frac{r}{r_s}\right)^2\right) \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Cored NFW Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`CoredNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)


class VikhlininDensityProfile(_RadialDensityProfile):
    r"""
    Vikhlinin Density Profile :footcite:p:`VikhlininProfile`.

    This profile is used to model the density of galaxy clusters, incorporating
    a truncation at large radii and additional flexibility for inner slopes.

    .. math::
        \rho(r) = \rho_0 \left(\frac{r}{r_c}\right)^{-0.5 \alpha}
        \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-1.5 \beta + 0.25 \alpha}
        \left(1 + \left(\frac{r}{r_s}\right)^{\gamma}\right)^{-0.5 \epsilon / \gamma}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_s` is the truncation radius.
    - :math:`\alpha, \beta, \gamma, \epsilon` control the slope and truncation behavior.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`VikhlininDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``r_s``
             - :math:`r_s`
             - Truncation radius
           * - ``alpha``
             - :math:`\alpha`
             - Controls the innermost slope
           * - ``beta``
             - :math:`\beta`
             - Governs the outer slope
           * - ``epsilon``
             - :math:`\epsilon`
             - Steepens the outer decline
           * - ``gamma``
             - :math:`\gamma`
             - Exponent in the truncation factor



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`VikhlininDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_c": unyt_quantity(1.0, "pc"),
        "r_s": unyt_quantity(1.0, "pc"),
        "alpha": unyt_quantity(1.0, ""),
        "beta": unyt_quantity(1.0, ""),
        "epsilon": unyt_quantity(1.0, ""),
        "gamma": unyt_quantity(3.0, ""),
    }

    @staticmethod
    def _profile(
        r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0
    ):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )


class AM06DensityProfile(_RadialDensityProfile):
    r"""
    Ascasibar and Markevitch (2006) Density Profile :footcite:p:`AM06Profile`.

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

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`AM06DensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``a_c``
             - :math:`a_c`
             - Core radius
           * - ``c``
             - :math:`c`
             - Concentration parameter
           * - ``a``
             - :math:`a`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Controls slope near the core
           * - ``beta``
             - :math:`\beta`
             - Controls outer slope



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`AM06DensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    Use Case
    --------
    This profile is well-suited for modeling dark matter halos with detailed inner and outer slope behaviors.


    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "a": unyt_quantity(1.0, "pc"),
        "a_c": unyt_quantity(1.0, "pc"),
        "c": unyt_quantity(1.0, ""),
        "alpha": unyt_quantity(1.0, ""),
        "beta": unyt_quantity(1.0, ""),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, a=1.0, a_c=1.0, c=1.0, alpha=1.0, beta=1.0):
        return (
            rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta
        )


class SNFWDensityProfile(_RadialDensityProfile):
    r"""
    Simplified Navarro-Frenk-White (SNFW) Density Profile :footcite:p:`SNFWProfile`.

    This profile is a simplified version of the NFW profile, widely used for modeling
    dark matter halos with specific scaling.

    .. math::
        \rho(r) = \frac{3M}{16\pi a^3} \frac{1}{\frac{r}{a} \left(1 + \frac{r}{a}\right)^{2.5}}

    where:

    - :math:`M` is the total mass.
    - :math:`a` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`SNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``a``
             - :math:`a`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`SNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base (no direct mass/potential expression coded here)



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "M": unyt_quantity(1.0, "Msun"),
        "a": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, M=1.0, a=1.0):
        return 3.0 * M / (16.0 * sp.pi * a**3) / ((r / a) * (1.0 + r / a) ** 2.5)


class TNFWDensityProfile(_RadialDensityProfile):
    r"""
    Truncated Navarro-Frenk-White (TNFW) Density Profile :footcite:p:`TNFWProfile`.

    This profile is a modification of the NFW profile with an additional truncation
    term to account for finite halo sizes.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}
        \frac{1}{1 + \left(\frac{r}{r_t}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`r_t` is the truncation radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`TNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``r_t``
             - :math:`r_t`
             - Truncation radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`TNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

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
    _is_parent_profile = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.

    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt_quantity(1.0, "pc"),
        "r_t": unyt_quantity(1.0, "pc"),
    }

    @staticmethod
    def _profile(r, rho_0=1.0, r_s=1.0, r_t=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2) / (1 + (r / r_t) ** 2)


class _CylindricalDensityProfile(CylindricalProfile, ABC):
    r"""
    Base class for cylindrical density profiles. This class provides a location in which to define
    various derived quantities which are shared across all the cylindrical profiles.
    """
    _is_parent_profile = True


class DoubleExponentialDisk(_CylindricalDensityProfile):
    r"""
    Double-Exponential Disk Density Profile :footcite:p:`DoubleExpProfile`.

    This profile models the density distribution of galactic disks using a double-exponential
    form, where the density falls off exponentially both radially and vertically.

    .. math::
        \rho(R, z) = \rho_0 \exp\left(-\frac{R}{h_r} - \frac{|z|}{h_z}\right)

    where:

    - :math:`\rho_0` is the central density.
    - :math:`h_r` is the radial scale length.
    - :math:`h_z` is the vertical scale height.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`DoubleExponentialDisk`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``h_r``
             - :math:`h_r`
             - Radial scale length
           * - ``h_z``
             - :math:`h_z`
             - Vertical scale height

    Use Case
    --------
    This profile is widely used to model the mass distribution of spiral galaxy disks,
    particularly in edge-on systems where vertical structure is visible.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import DoubleExponentialDisk

        >>> R = np.linspace(0.1, 10, 100)
        >>> Z = 0  # Mid-plane slice
        >>> profile = DoubleExponentialDisk(rho_0=1.0, h_r=2.5, h_z=0.3)
        >>> rho = profile(R, Z)

        >>> _ = plt.semilogy(R, rho, 'k-', label='Double-Exponential Disk')
        >>> _ = plt.xlabel('Radius (R)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    MiyamotoNagaiDisk
    """
    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "rho_0": unyt_quantity(1.0, "Msun/pc**3"),
        "h_r": unyt_quantity(1.0, "pc"),
        "h_z": unyt_quantity(1.0, "pc"),
    }

    def _set_output_units(self):
        return self._parameters["rho_0"].units

    @staticmethod
    def _profile(r, z, rho_0=1.0, h_r=1.0, h_z=1.0):
        return rho_0 * sp.exp(-(r / h_r) - (sp.Abs(z) / h_z))


class MiyamotoNagaiDisk(_CylindricalDensityProfile):
    r"""
    Miyamoto-Nagai Disk Density Profile :footcite:p:`MiyamotoNagaiProfile`.

    This profile is an axisymmetric potential-density model that smoothly transitions between
    a disk-like and spherical shape. It is commonly used in dynamical models of galaxies.

    .. math::
        \rho(R, z) = \frac{b^2 M}{4\pi} \frac{(a R^2) + (3\sqrt{z^2 + b^2} + a)(\sqrt{z^2 + b^2} + a)^2}
        {(R^2 + (\sqrt{z^2 + b^2} + a)^2)^{5/2} (z^2 + b^2)^{3/2}}

    where:

    - :math:`M` is the total mass of the disk.
    - :math:`a` is the radial scale length.
    - :math:`b` is the vertical scale height.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`MiyamotoNagaiDisk`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``a``
             - :math:`a`
             - Radial scale length
           * - ``b``
             - :math:`b`
             - Vertical scale height

    Use Case
    --------
    The Miyamoto-Nagai disk is widely used in galactic dynamics because it provides
    an analytic potential that smoothly interpolates between a thick disk and a
    quasi-spherical mass distribution.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import MiyamotoNagaiDisk

        >>> R = np.linspace(0.1, 10, 100)
        >>> Z = 0  # Mid-plane slice
        >>> profile = MiyamotoNagaiDisk(M=1.0, a=3.0, b=0.5)
        >>> rho = profile(R, Z)

        >>> _ = plt.semilogy(R, rho, 'k-', label='Miyamoto-Nagai Disk')
        >>> _ = plt.xlabel('Radius (R)')
        >>> _ = plt.ylabel('Density (rho)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    DoubleExponentialDisk
    """
    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "M": unyt_quantity(1.0, "Msun"),
        "a": unyt_quantity(1.0, "pc"),
        "b": unyt_quantity(1.0, "pc"),
    }

    def _set_output_units(self):
        return self._parameters["M"].units / self.axes_units["r"] ** 3

    @staticmethod
    def _profile(r, z, M=1.0, a=1.0, b=1.0):
        coeff = b**2 * M / (4 * sp.pi)
        num = (a * r**2) + (3 * sp.sqrt(z**2 + b**2) + a) * (
            sp.sqrt(z**2 + b**2) + a
        ) ** 2
        den = (r**2 + (sp.sqrt(z**2 + b**2) + a) ** 2) ** (5 / 2) * (
            z**2 + b**2
        ) ** (3 / 2)
        return coeff * num / den
