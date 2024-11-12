import numpy as np

from profiles.abc import FixedProfile


class RadialEntropyProfile(FixedProfile):
    axes = ["r"]
    units = "keV*cm^2"


class BaselineEntropyProfile(RadialEntropyProfile):
    PARAMETERS = {
        "K_0": 10.0,
        "K_200": 200.0,
        "r_200": 1000.0,
        "alpha": 1.1,
    }

    @staticmethod
    def FUNCTION(r, K_0, K_200, r_200, alpha):
        return K_0 + K_200 * (r / r_200) ** alpha


class BrokenEntropyProfile(RadialEntropyProfile):
    PARAMETERS = {
        "r_s": 300.0,
        "K_scale": 200.0,
        "alpha": 1.1,
        "K_0": 0.0,
    }

    @staticmethod
    def FUNCTION(r, r_s, K_scale, alpha, K_0=0.0):
        x = r / r_s
        ret = (x**alpha) * (1.0 + x**5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)


class WalkerEntropyProfile(RadialEntropyProfile):
    PARAMETERS = {
        "r_200": 1000.0,
        "A": 0.5,
        "B": 0.2,
        "K_scale": 100.0,
        "alpha": 1.1,
    }

    @staticmethod
    def FUNCTION(r, r_200, A, B, K_scale, alpha=1.1):
        x = r / r_200
        return K_scale * (A * x**alpha) * np.exp(-((x / B) ** 2))
