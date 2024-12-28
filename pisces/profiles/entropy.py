"""
Entropy profiles for astrophysical modeling.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from pisces.profiles.base import Profile
import sympy as sp

class RadialEntropyProfile(Profile):
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES =  ['r']
    PARAMETERS = None
    UNITS: str = "keV*cm^2"


class BaselineEntropyProfile(RadialEntropyProfile):
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "K_0": 10.0,
        "K_200": 200.0,
        "r_200": 1000.0,
        "alpha": 1.1,
    }

    @staticmethod
    def _function(r, K_0, K_200, r_200, alpha):
        return K_0 + K_200 * (r / r_200) ** alpha


class BrokenEntropyProfile(RadialEntropyProfile):
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "r_s": 300.0,
        "K_scale": 200.0,
        "alpha": 1.1,
        "K_0": 0.0,
    }

    @staticmethod
    def _function(r, r_s, K_scale, alpha, K_0=0.0):
        x = r / r_s
        ret = (x**alpha) * (1.0 + x**5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)


class WalkerEntropyProfile(RadialEntropyProfile):
    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    # Generally, these do not need to be changed in subclasses; however, they
    # may be if necessary. Ensure that any metaclasses / ABC's have _IS_ABC=True.
    _IS_ABC = False

    # @@ CLASS ATTRIBUTES @@ #
    # These attributes should be set / manipulated in all subclasses to
    # implement the desired behavior.
    AXES: List[str] = ['r']
    PARAMETERS: Dict[str, Any] = {
        "r_200": 1000.0,
        "A": 0.5,
        "B": 0.2,
        "K_scale": 100.0,
        "alpha": 1.1,
    }

    @staticmethod
    def _function(r, r_200, A, B, K_scale, alpha=1.1):
        x = r / r_200
        return K_scale * (A * x**alpha) * sp.exp(-((x / B) ** 2))
