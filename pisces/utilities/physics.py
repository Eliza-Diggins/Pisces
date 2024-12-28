"""
Utilities for basic physics and constants.

The :py:mod:`pisces.utilities.physics` module provides access to physical constants and
other utilities for computing basic physical quantities like mean molecular weights and gas fractions.
"""
from unyt import physical_constants as pc
from unyt import unyt_quantity
from pisces.utilities.config import pisces_params

# Physical constants
m_p: unyt_quantity = pc.mp
"""unyt_quantity: Proton mass."""
G = pc.G
"""unyt_quantity: Gravitational constant."""
kboltz = pc.kboltz
"""unyt_quantity: Boltzmann constant."""

def compute_mean_molecular_weight(hydrogen_fraction: float):
    r"""
    Compute the mean molecular weight for a given primordial Hydrogen fraction assuming all non-hydrogen species
    are Helium.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight is the mass per particle in a fluid. Thus,

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{\sum_k n_k + n_{k,e^-}} \approx \frac{\sum_k n_k m_k}{\sum_k n_k + n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{2n_{\rm H} + 3n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{2\chi_H + (3/4)(1-\chi_H)}.

    """
    return 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))

def compute_mean_molecular_weight_per_electron(hydrogen_fraction: float):
    """
    Compute the mean molecular weight (:math:`\mu`) per free electron for a given primordial Hydrogen fraction.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight per electron is

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{n_{k,e^-}} \approx \frac{\sum_k n_k m_k}{n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{1n_{\rm H} + 2n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{1\chi_H + (1/2)(1-\chi_H)}.

    """
    return 1/(hydrogen_fraction + 0.5*(1-hydrogen_fraction))

# Gas properties
X_H = pisces_params['physics.hydrogen_abundance']  # Hydrogen mass fraction (primordial gas)
r"""
The default hydrogen mass fraction, :math:`\chi_H`, which is typical for primordial gas based on cosmological observations.

The :math:`\chi_H` value can be set in the Pisces configuration.
"""

mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))  # Mean molecular weight
""" The mean molecular weight (:math:`\mu`) using the default hydrogen abundance.
"""

mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))  # Mean molecular weight per free electron
""" The mean molecular weight per electron (:math:`\mu_e`) using the default hydrogen abundance.
"""
