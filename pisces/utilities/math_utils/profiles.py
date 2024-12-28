""" Mathematical utilities for profiles in
various contexts.
"""
from typing import Callable

def build_asymptotic_extension(function: Callable, r0: float, n: float) -> Callable:
    r"""
    Create an asymptotic extension of a given function.

    This utility generates a callable function that extends the behavior of the input `function`
    beyond a specified boundary :math:`r_0` using a power-law approximation:

    .. math::

        f(r) = f(r_0) \left(\frac{r}{r_0}\right)^n \quad \text{for } r > r_0.

    Parameters
    ----------
    function : Callable
        The original function to extend asymptotically. Must be a callable that takes a single argument (radius)
        and returns a scalar value.
    r0 : float
        The boundary radius where the asymptotic behavior begins.
    n : float
        The power-law index that describes the asymptotic behavior of the function. Typically, :math:`n < 0`
        for decreasing functions.

    Returns
    -------
    Callable
        A new function that matches the input `function` for :math:`r = r_0` and extends it asymptotically
        for :math:`r > r_0`.

    Notes
    -----
    - This utility is useful for extending density profiles, potential functions, or other radial functions
      when their asymptotic behavior can be approximated by a power law.
    - The input ``function`` is evaluated once at :math:`r_0` to determine the boundary value.
    """
    # Evaluate the function at the boundary radius r0
    _boundary_value = function(r0)

    # Return the asymptotic extension as a lambda function
    return lambda _r: _boundary_value * (_r / r0)**n
