"""
Utilities for symbolic mathematical manipulations.

This module relies on the `Sympy Package <https://sympy.org>`_ to perform various mathematical operations. The ``sympy``
module provides important backends for both the differential geometry of the :py:mod:`~pisces.geometry` module and the
:py:mod:`~pisces.profiles` module.
"""
import sympy as sp

def get_powerlaw_limit(expression, variable, limit: str = 'outer'):
    """
    Extract the coefficient and power of the dominant power-law term
    in a given expression at small (inner) or large (outer) limits.

    Parameters
    ----------
    expression : str or sp.Basic
        The mathematical expression to analyze.
    variable : str or sp.Symbol
        The variable with respect to which the limit is taken.
    limit : str, optional
        Either 'inner' (r -> 0) or 'outer' (r -> infinity), default is 'outer'.

    Returns
    -------
    Tuple[sp.Basic, sp.Basic]
        A tuple containing the coefficient and the power of the dominant term.
    """
    # Ensure inputs are SymPy-compatible
    expression = sp.sympify(expression)
    if isinstance(variable, str):
        variable = sp.Symbol(variable)

    # Validate the limit
    if limit == 'outer':
        series_point = sp.oo
    elif limit == 'inner':
        series_point = 0
    else:
        raise ValueError("Invalid limit. Must be 'outer' or 'inner'.")

    # Expand the series around the desired point

    try:
        expansion = sp.series(expression, variable, series_point).removeO()
    except Exception as e:
        raise ValueError(f"Failed to expand the expression: {e}")

    # Construct the ordered terms and fix any issues with the symbols.
    # In many cases, the base ends up being (1/x), which we don't want.
    terms = [_term.as_coeff_mul(variable) for _term in expansion.as_ordered_terms()]
    terms = [(_term[0],_term[1][0].as_base_exp()) if len(_term[1]) else (_term[0],(variable,0)) for _term in terms]
    terms = [(_term[0],_term[1][1]) if _term[1][0] == variable else (_term[0],-_term[1][1]) for _term in terms]
    terms = sorted(terms, key=lambda x: x[1])

    if limit == 'outer':
        critical_term = terms[-1]
    else:
        critical_term = terms[0]

    return critical_term
