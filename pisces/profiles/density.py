from pisces.profiles.base import FixedProfile
import numpy as np

class RadialDensityProfile(FixedProfile):
    """
    Base class for radial density profiles with fixed axes, units, and parameters.
    """
    AXES = ["r"]
    UNITS = "Msun/kpc**3"

class NFWDensityProfile(RadialDensityProfile):
    """
    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> q = NFWDensityProfile(rho_0=1,r_s=1)
    >>> x = np.linspace(1,100,1000)
    >>> y = q(x)
    >>> plt.plot(x,y) # doctest: +SKIP
    >>> plt.show()

    """
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

class HernquistDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)


class EinastoDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
        "alpha": 0.18,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0, alpha=0.18):
        return rho_0 * np.exp(-2 * alpha * ((r / r_s) ** alpha - 1))

class SingularIsothermalDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0):
        return rho_0 / r**2

class CoredIsothermalDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_c": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

class PlummerDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "M": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, M=1.0, r_s=1.0):
        return (3 * M) / (4 * np.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

class DehnenDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "M": 1.0,
        "r_s": 1.0,
        "gamma": 1.0,
    }

    @staticmethod
    def FUNCTION(r, M=1.0, r_s=1.0, gamma=1.0):
        return (
            ((3 - gamma) * M)
            / (4 * np.pi * r_s**3)
            * (r / r_s) ** (-gamma)
            * (1 + r / r_s) ** (gamma - 4)
        )

class JaffeDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)


class KingDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_c": 1.0,
        "r_t": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_c=1.0, r_t=1.0):
        return rho_0 * ((1 + (r / r_c) ** 2) ** (-3 / 2) - (1 + (r_t / r_c) ** 2) ** (-3 / 2))


class BurkertDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))


class MooreDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))


class CoredNFWDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)


class VikhlininDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_c": 1.0,
        "r_s": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "epsilon": 1.0,
        "gamma": 3.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )


class AM06DensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "a": 1.0,
        "a_c": 1.0,
        "c": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, a=1.0, a_c=1.0, c=1.0, alpha=1.0, beta=1.0):
        return rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta


class SNFWDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "M": 1.0,
        "a": 1.0,
    }

    @staticmethod
    def FUNCTION(r, M=1.0, a=1.0):
        return 3.0 * M / (16.0 * np.pi * a**3) / ((r / a) * (1.0 + r / a) ** 2.5)


class TNFWDensityProfile(RadialDensityProfile):
    PARAMETERS = {
        "rho_0": 1.0,
        "r_s": 1.0,
        "r_t": 1.0,
    }

    @staticmethod
    def FUNCTION(r, rho_0=1.0, r_s=1.0, r_t=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2) / (1 + (r / r_t) ** 2)
