"""
Entropy profiles for astrophysical modeling.
"""
from typing import List, Dict, Any

import numpy as np
import sympy as sp

from pisces.profiles.base import Profile


class RadialEntropyProfile(Profile):
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES = ['r']
    DEFAULT_PARAMETERS = None
    DEFAULT_UNITS: str = "keV*cm^2"


class BaselineEntropyProfile(RadialEntropyProfile):
    r"""
    Baseline Entropy Profile :footcite:p:`BaselineEntropy`.

    This profile represents a simple power-law–type baseline for the entropy
    distribution in a hot gaseous halo or galaxy cluster. It allows for a
    central “floor” of entropy (:math:`K_0`) plus a power-law rise with radius.

    .. math::
        K(r) = K_0 + K_{200} \left(\frac{r}{r_{200}}\right)^{\alpha}

    where:

    - :math:`K_0` is the constant (baseline) entropy offset in the center.
    - :math:`K_{200}` is the normalization of the power-law component.
    - :math:`r_{200}` is a characteristic radius (e.g., related to :math:`R_{200}` of the system).
    - :math:`\alpha` controls how steeply the entropy rises with radius.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.entropy import BaselineEntropyProfile

        >>> r = np.linspace(10, 1000, 100)
        >>> profile = BaselineEntropyProfile(K_0=10.0, K_200=200.0, r_200=1000.0, alpha=1.1)
        >>> K = profile(r)

        >>> _ = plt.loglog(r, K, 'r-', label='Baseline Entropy Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Entropy (K)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    BrokenEntropyProfile, WalkerEntropyProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "K_0": 10.0,
        "K_200": 200.0,
        "r_200": 1000.0,
        "alpha": 1.1,
    }

    @staticmethod
    def _function(r, K_0, K_200, r_200, alpha):
        return K_0 + K_200 * (r / r_200) ** alpha


class BrokenEntropyProfile(RadialEntropyProfile):
    r"""
        Broken Entropy Profile :footcite:p:`BrokenEntropy`.

        This profile introduces a “broken” or piecewise-like scaling that transitions
        at larger radii. It starts as a power law but includes a high-radius modification
        governed by an additional factor :math:`(1 + x^5)^{0.2\,(\,1.1 - \alpha)}`. The presence
        of :math:`K_0` provides a possible core or offset.

        .. math::
            K(r) = K_\mathrm{scale}\,\bigl[K_0 + (x)^{\alpha}
                   \bigl(1 + x^{5}\bigr)^{0.2\,(1.1 - \alpha)}\bigr],
            \quad x = \frac{r}{r_s}

        where:

        - :math:`r_s` is a scale radius.
        - :math:`K_\mathrm{scale}` sets the overall normalization.
        - :math:`\alpha` is the power-law slope at intermediate radii.
        - :math:`K_0` is an added offset term, allowing for a central floor of entropy.

        References
        ----------
        .. footbibliography::

        Example
        -------
        .. plot::
            :include-source:

            >>> import numpy as np
            >>> import matplotlib.pyplot as plt
            >>> from pisces.profiles.entropy import BrokenEntropyProfile

            >>> r = np.logspace(1, 3, 100)
            >>> profile = BrokenEntropyProfile(r_s=300.0, K_scale=200.0, alpha=1.1, K_0=0.0)
            >>> K = profile(r)

            >>> _ = plt.loglog(r, K, 'b-', label='Broken Entropy Profile')
            >>> _ = plt.xlabel('Radius (r)')
            >>> _ = plt.ylabel('Entropy (K)')
            >>> _ = plt.legend()
            >>> plt.show()

        See Also
        --------
        BaselineEntropyProfile, WalkerEntropyProfile
        """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "r_s": 300.0,
        "K_scale": 200.0,
        "alpha": 1.1,
        "K_0": 0.0,
    }

    @staticmethod
    def _function(r, r_s, K_scale, alpha, K_0=0.0):
        x = r / r_s
        ret = (x ** alpha) * (1.0 + x ** 5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)


class WalkerEntropyProfile(RadialEntropyProfile):
    r"""
    Walker Entropy Profile :footcite:p:`WalkerEntropy`.

    This profile is inspired by a Gaussian‐truncated power law, where the factor
    :math:`\exp\bigl[-(x/B)^2\bigr]` modifies the growth at large radius. The user can
    adjust both the amplitude (:math:`K_\mathrm{scale}`) and the slope (:math:`\alpha`).

    .. math::
        K(r) = K_\mathrm{scale} \,\Bigl(A\,x^{\alpha}\Bigr)\,
               \exp\Bigl[-\bigl(\tfrac{x}{B}\bigr)^{2}\Bigr],
        \quad x = \frac{r}{r_{200}},

    where:

    - :math:`r_{200}` is a characteristic radius or virial scale.
    - :math:`A` and :math:`B` shape the amplitude and radial cutoff.
    - :math:`K_\mathrm{scale}` sets the overall normalization.
    - :math:`\alpha` is the power-law slope before the exponential cutoff.

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.entropy import WalkerEntropyProfile

        >>> r = np.linspace(0, 2000, 200)
        >>> profile = WalkerEntropyProfile(r_200=1000.0, A=0.5, B=0.2, K_scale=100.0, alpha=1.1)
        >>> K = profile(r)

        >>> _ = plt.semilogy(r, K, 'g-', label='Walker Entropy Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Entropy (K)')
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    BaselineEntropyProfile, BrokenEntropyProfile
    """
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    DEFAULT_PARAMETERS: Dict[str, Any] = {
        "r_200": 1000.0,
        "A": 0.5,
        "B": 0.2,
        "K_scale": 100.0,
        "alpha": 1.1,
    }

    @staticmethod
    def _function(r, r_200, A, B, K_scale, alpha=1.1):
        x = r / r_200
        return K_scale * (A * x ** alpha) * sp.exp(-((x / B) ** 2))
