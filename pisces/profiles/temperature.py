"""
Temperature profiles for astrophysical modeling.
"""
from pisces.profiles.base import Profile
from typing import Dict, Any, List
import sympy as sp

class RadialTemperatureProfile(Profile):
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES =  ['r']
    PARAMETERS = None
    UNITS: str = "keV"

class VikhlininTemperatureProfile(RadialTemperatureProfile):
    r"""
    Vikhlinin Temperature Profile.

    This profile models the temperature distribution in galaxy clusters, incorporating both
    a core and a declining profile at large radii. It also accounts for a potential cooling region.

    .. math::
        T(r) = T_0 \cdot \left( \frac{r}{r_t} \right)^{-a} \left(1 + \left( \frac{r}{r_t} \right)^b \right)^{-c/b}
               \cdot \frac{\left(\left(\frac{r}{r_\text{cool}}\right)^{a_\text{cool}} + \frac{T_\text{min}}{T_0}\right)}
                      {\left(\left(\frac{r}{r_\text{cool}}\right)^{a_\text{cool}} + 1\right)}

    where:

    - :math:`T_0` is the central temperature.
    - :math:`r_t` is the truncation radius.
    - :math:`a, b, c` control the slope and truncation.
    - :math:`T_\text{min}` is the minimum temperature in the cooling region.
    - :math:`r_\text{cool}, a_\text{cool}` describe the cooling region.

    References
    ----------
    .. [Vik06] Vikhlinin et al., 2006.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import VikhlininTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = VikhlininTemperatureProfile(
        ...     T_0=5.0, a=-0.1, b=2, c=1.2, r_t=100.0, T_min=2.0, r_cool=10.0, a_cool=-0.2
        ... )
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'b-', label='Vikhlinin Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    BetaModelTemperatureProfile, DoubleBetaTemperatureProfile, AM06TemperatureProfile
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
        "T_0": 5.0,
        "a": 0.1,
        "b": 0.5,
        "c": 1.2,
        "r_t": 200.0,
        "T_min": 2.0,
        "r_cool": 10.0,
        "a_cool": 0.2,
    }

    @staticmethod
    def _function(r, T_0, a, b, c, r_t, T_min, r_cool, a_cool):
        x = (r / r_cool) ** a_cool
        t = (r / r_t) ** (-a) / ((1.0 + (r / r_t) ** b) ** (c / b))
        return T_0 * t * (x + T_min / T_0) / (x + 1)


class AM06TemperatureProfile(RadialTemperatureProfile):
    r"""
    An & Zhao (2006) Temperature Profile (AM06).

    This profile describes a general temperature distribution for galaxy clusters with flexible
    inner and outer slopes.

    .. math::
        T(r) = \frac{T_0}{1 + \frac{r}{a}} \cdot \frac{c + \frac{r}{a_c}}{1 + \frac{r}{a_c}}

    where:

    - :math:`T_0` is the central temperature.
    - :math:`a` is the scale radius.
    - :math:`a_c` is the core radius.
    - :math:`c` modifies the slope of the inner region.

    References
    ----------
    .. [An06] An, J., & Zhao, H. S., 2006.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import AM06TemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = AM06TemperatureProfile(T_0=4.0, a=300.0, a_c=50.0, c=0.2)
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'g-', label='AM06 Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    VikhlininTemperatureProfile, BetaModelTemperatureProfile, UniversalPressureTemperatureProfile
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
        "T_0": 4.0,
        "a": 300.0,
        "a_c": 50.0,
        "c": 0.2,
    }

    @staticmethod
    def _function(r, T_0, a, a_c, c):
        return T_0 / (1.0 + r / a) * (c + r / a_c) / (1.0 + r / a_c)


class UniversalPressureTemperatureProfile(RadialTemperatureProfile):
    r"""
    Universal Pressure Temperature Profile.

    This profile assumes a temperature distribution inversely proportional to the square root
    of the pressure profile.

    .. math::
        T(r) = T_0 \cdot \left(1 + \frac{r}{r_s}\right)^{-1.5}

    where:

    - :math:`T_0` is the central temperature.
    - :math:`r_s` is the scale radius.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import UniversalPressureTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = UniversalPressureTemperatureProfile(T_0=5.0, r_s=300.0)
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'r-', label='Universal Pressure Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    AM06TemperatureProfile, IsothermalTemperatureProfile, BetaModelTemperatureProfile
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
        "T_0": 5.0,
        "r_s": 300.0,
    }

    @staticmethod
    def _function(r, T_0, r_s):
        return T_0 * (1 + r / r_s) ** -1.5


class IsothermalTemperatureProfile(RadialTemperatureProfile):
    r"""
    Isothermal Temperature Profile.

    This profile models a constant temperature throughout the radius of the system.

    .. math::
        T(r) = T_0

    where:

    - :math:`T_0` is the constant temperature.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import IsothermalTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = IsothermalTemperatureProfile(T_0=5.0)
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'c-', label='Isothermal Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    UniversalPressureTemperatureProfile, BetaModelTemperatureProfile, CoolingFlowTemperatureProfile
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
        "T_0": 5.0,
    }

    @staticmethod
    def _function(r, T_0):
        return T_0


class CoolingFlowTemperatureProfile(RadialTemperatureProfile):
    r"""
    Cooling Flow Temperature Profile.

    This profile models the temperature decline in regions experiencing significant cooling flows.

    .. math::
        T(r) = T_0 \cdot \left( \frac{r}{r_c} \right)^{-a}

    where:

    - :math:`T_0` is the central temperature.
    - :math:`r_c` is the core radius.
    - :math:`a` controls the slope of the temperature decline.


    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import CoolingFlowTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = CoolingFlowTemperatureProfile(T_0=5.0, r_c=100.0, a=0.8)
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'm-', label='Cooling Flow Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    VikhlininTemperatureProfile, BetaModelTemperatureProfile, IsothermalTemperatureProfile
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
        "T_0": 5.0,
        "r_c": 100.0,
        "a": 0.8,
    }

    @staticmethod
    def _function(r, T_0, r_c, a):
        return T_0 * (r / r_c) ** -a


class DoubleBetaTemperatureProfile(RadialTemperatureProfile):
    r"""
    Double Beta Temperature Profile.

    This profile combines two beta models to represent systems with two distinct temperature components.

    .. math::
        T(r) = T_0 \cdot \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta_1}
               + T_1 \cdot \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta_2}

    where:

    - :math:`T_0` is the central temperature of the first component.
    - :math:`r_c` is the core radius.
    - :math:`\beta_1` controls the slope of the first component.
    - :math:`T_1` is the central temperature of the second component.
    - :math:`\beta_2` controls the slope of the second component.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import DoubleBetaTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = DoubleBetaTemperatureProfile(
        ...     T_0=5.0, r_c=100.0, beta_1=0.8, T_1=3.0, beta_2=1.2
        ... )
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'c-', label='Double Beta Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    BetaModelTemperatureProfile, VikhlininTemperatureProfile, AM06TemperatureProfile
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
        "T_0": 5.0,
        "r_c": 100.0,
        "beta_1": 0.8,
        "T_1": 3.0,
        "beta_2": 1.2,
    }

    @staticmethod
    def _function(r, T_0, r_c, beta_1, T_1, beta_2):
        return T_0 * (1 + (r / r_c) ** 2) ** -beta_1 + T_1 * (1 + (r / r_c) ** 2) ** -beta_2


class BetaModelTemperatureProfile(RadialTemperatureProfile):
    r"""
    Beta Model Temperature Profile.

    This profile describes temperature distributions with a single beta model.

    .. math::
        T(r) = T_0 \cdot \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta}

    where:

    - :math:`T_0` is the central temperature.
    - :math:`r_c` is the core radius.
    - :math:`\beta` controls the slope of the temperature decline.

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.temperature import BetaModelTemperatureProfile

        >>> r = np.logspace(-1, 2, 100)
        >>> profile = BetaModelTemperatureProfile(T_0=5.0, r_c=100.0, beta=0.8)
        >>> T = profile(r)

        >>> _ = plt.semilogx(r, T, 'k-', label='Beta Model Temperature Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Temperature (T)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    DoubleBetaTemperatureProfile, VikhlininTemperatureProfile, CoolingFlowTemperatureProfile
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
        "T_0": 5.0,
        "r_c": 100.0,
        "beta": 0.8,
    }

    @staticmethod
    def _function(r, T_0, r_c, beta):
        return T_0 * (1 + (r / r_c) ** 2) ** -beta


