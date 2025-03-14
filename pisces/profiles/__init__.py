"""
Physical profiles for modeling in Pisces.

Profiles in Pisces
==================

Profiles are one of the core components of Pisces, enabling users to model and manipulate mathematical profiles with
symbolic and numerical capabilities. A profile represents a mathematical function or relationship, parameterized by variables
(:py:attr:`~pisces.profiles.base.Profile.AXES`) and constants (:py:attr:`~pisces.profiles.base.Profile.DEFAULT_PARAMETERS`).
Profiles in Pisces are highly flexible, supporting:

- Symbolic definitions using `SymPy <https://www.sympy.org>`_ for analytical manipulation.
- Numerical evaluation using `NumPy <https://numpy.org>`_ for performance.
- Units handling via `unyt <https://unyt.readthedocs.io>`_ for physical consistency.
- Extensibility through derived attributes and user-defined expressions.

Profiles can represent various physical and mathematical entities, such as density, mass, or temperature distributions.

"""
from .base import Profile
from .density import (
    AM06DensityProfile,
    BurkertDensityProfile,
    CoredIsothermalDensityProfile,
    CoredNFWDensityProfile,
    DehnenDensityProfile,
    EinastoDensityProfile,
    HernquistDensityProfile,
    JaffeDensityProfile,
    KingDensityProfile,
    MooreDensityProfile,
    NFWDensityProfile,
    PlummerDensityProfile,
    SingularIsothermalDensityProfile,
    SNFWDensityProfile,
    TNFWDensityProfile,
    VikhlininDensityProfile,
    _RadialDensityProfile,
)
from .entropy import (
    BaselineEntropyProfile,
    BrokenEntropyProfile,
    WalkerEntropyProfile,
    _RadialEntropyProfile,
)
from .temperature import (
    AM06TemperatureProfile,
    BetaModelTemperatureProfile,
    CoolingFlowTemperatureProfile,
    DoubleBetaTemperatureProfile,
    IsothermalTemperatureProfile,
    UniversalPressureTemperatureProfile,
    VikhlininTemperatureProfile,
    _RadialTemperatureProfile,
)
