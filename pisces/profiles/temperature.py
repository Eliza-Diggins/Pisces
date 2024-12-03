from pisces.profiles.base import FixedProfile
import numpy as np

class RadialTemperatureProfile(FixedProfile):
    AXES = ["r"]
    UNITS = "keV"


class VikhlininTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
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
    def FUNCTION(r, T_0, a, b, c, r_t, T_min, r_cool, a_cool):
        x = (r / r_cool) ** a_cool
        t = (r / r_t) ** (-a) / ((1.0 + (r / r_t) ** b) ** (c / b))
        return T_0 * t * (x + T_min / T_0) / (x + 1)


class AM06TemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 4.0,
        "a": 300.0,
        "a_c": 50.0,
        "c": 0.2,
    }

    @staticmethod
    def FUNCTION(r, T_0, a, a_c, c):
        return T_0 / (1.0 + r / a) * (c + r / a_c) / (1.0 + r / a_c)


class UniversalPressureTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 5.0,
        "r_s": 300.0,
    }

    @staticmethod
    def FUNCTION(r, T_0, r_s):
        return T_0 * (1 + r / r_s) ** -1.5


class IsothermalTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 5.0,
    }

    @staticmethod
    def FUNCTION(r, T_0):
        return T_0*np.ones_like(r)


class CoolingFlowTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 5.0,
        "r_c": 100.0,
        "a": 0.8,
    }

    @staticmethod
    def FUNCTION(r, T_0, r_c, a):
        return T_0 * (r / r_c) ** -a


class DoubleBetaTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 5.0,
        "r_c": 100.0,
        "beta_1": 0.8,
        "T_1": 3.0,
        "beta_2": 1.2,
    }

    @staticmethod
    def FUNCTION(r, T_0, r_c, beta_1, T_1, beta_2):
        return T_0 * (1 + (r / r_c) ** 2) ** -beta_1 + T_1 * (1 + (r / r_c) ** 2) ** -beta_2


class BetaModelTemperatureProfile(RadialTemperatureProfile):
    PARAMETERS = {
        "T_0": 5.0,
        "r_c": 100.0,
        "beta": 0.8,
    }

    @staticmethod
    def FUNCTION(r, T_0, r_c, beta):
        return T_0 * (1 + (r / r_c) ** 2) ** -beta


