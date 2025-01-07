"""
Mathematics utilities for solving the `Poisson equation <https://en.wikipedia.org/wiki/Poisson%27s_equation>`_
in various special cases.

This module provides functions for solving the Poisson problem in some of the special cases encountered
frequently in Pisces (and in astrophysics in general). For the most part, these methods simplify to simple
quadrature; however, many cases involve considerable extra legwork to implement.

Functions in this module are designed for flexibility, supporting both callable and array-like inputs
for density profiles. Generally, all of these functions have higher level interfaces in the relevant coordinate
system classes.

For background on the various techniques employed in this module, we encourage you to read the :ref:`poisson_equation` page.

Notes
-----

- All computations assume natural (Planck) units for consistency.
- Users must ensure that density profiles are smooth and continuous for accurate results.

"""
from typing import TYPE_CHECKING, Callable, Tuple, Union

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.interpolate import InterpolatedUnivariateSpline

from pisces.utilities.array_utils import CoordinateArray
from pisces.utilities.math_utils.numeric import integrate, integrate_from_zero

if TYPE_CHECKING:
    from pisces.geometry.coordinate_systems import PseudoSphericalCoordinateSystem


def _compute_spherical_poisson_boundary_integrated(
    density_profile: Callable, r0: float
) -> float:
    r"""
    Compute the outer boundary integral for the spherical Poisson problem using numerical integration.

    This method computes the contribution to the gravitational potential from mass at radii greater
    than ``r0``, assuming the density profile is well-defined and smooth at all radii (including out to infinity).
    This method is NOT suitable for use with splines.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\rho(r)`, which must be callable.
    r0 : float
        The radius beyond which the integral is computed.

    Returns
    -------
    float
        The computed integral value.

    Raises
    ------
    ValueError
        If ``density_profile`` is not callable.

    Notes
    -----
    The integral is computed as:

    .. math::

        I = \int_{r_0}^\infty r \rho(r) \, dr.

    Accurate results depend on the density profile decaying sufficiently fast at large radii.
    """
    if not callable(density_profile):
        raise ValueError(
            f"`density_profile` must be callable, not {type(density_profile)}."
        )

    # Define the integrand and compute the integral
    integrand = lambda r: density_profile(r) * r
    return quad(integrand, r0, np.inf)[0]


def _compute_spherical_poisson_boundary_asymptotic(
    density_profile: Callable, r0: float, n: int = -3
) -> float:
    r"""
    Compute the outer boundary integral for the spherical Poisson problem using an asymptotic power-law approximation.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\\rho(r)`, which must be callable.
    r0 : float
        The radius beyond which the asymptotic approximation is applied.
    n : int
        The power-law index :math:`n` for the asymptotic density profile. Default is ``-3``.

    Returns
    -------
    float
        The computed asymptotic integral value.

    Raises
    ------
    ValueError
        If the power-law index ``n`` is greater than or equal to ``-2`` (causing divergence).

    Notes
    -----
    The density profile is assumed to behave as:

    .. math::
        \\rho(r) \\sim \\rho(r_0) \\left(\\frac{r}{r_0}\\right)^n.

    The integral is approximated as:

    .. math::
        I \\approx -\\frac{\\rho(r_0) r_0^2}{n + 2},

    provided :math:`n < -2` for convergence.
    """
    if n >= -2:
        raise ValueError(
            "Boundary behavior (`n`, power index) must be less than -2 for convergence."
        )

    rho_at_radius = density_profile(r0)
    return -rho_at_radius * r0**2 / (n + 2)


def solve_poisson_spherical(
    density_profile: Union[np.ndarray, Callable],
    coordinates: np.ndarray,
    /,
    powerlaw_index: int = None,
    *,
    boundary_mode: str = "asymptotic",
) -> np.ndarray:
    r"""
    Solve Poisson's equation for a spherically symmetric distribution of mass specified by :math:`\rho(r)`.

    This function computes the gravitational potential :math:`\Phi(r)` for a radial density profile
    :math:`\rho(r)` in spherical coordinates by solving Poisson's equation:

    .. math::
        \nabla^2 \Phi = 4 \pi \rho(r).

    Parameters
    ----------
    density_profile : np.ndarray or Callable
        The density profile of the system. Can be provided as:

        - A callable function that takes a single argument (radius) and returns the density.
        - A numerical array of density values corresponding to the radii specified in ``coordinates``.
    coordinates : np.ndarray
        Array of radii (in ascending order) where the potential is computed.
    boundary_mode : str, optional
        Method for handling the outer boundary condition. Options are:

        - ``'integrate'``: Numerically integrates the density profile to infinity.
        - ``'asymptotic'``: Approximates the boundary contribution using an asymptotic power-law behavior.
          Default is ``'asymptotic'``.

    powerlaw_index : int, optional
        Power-law index for the asymptotic density behavior. Required when ``boundary_mode`` is ``'asymptotic'``.
        Default is ``None``, which will raise an error if it is needed and not specified.

    Returns
    -------
    np.ndarray
        The computed gravitational potential values at each radius in ``coordinates``.

    Raises
    ------
    ValueError
        If ``boundary_mode`` is not ``'integrate'`` or ``'asymptotic'``.

    Notes
    -----
    The potential is computed as :footcite:p:`BovyGalaxyBook` :

    .. math::
        \Phi(r) = -4 \pi \left[\frac{1}{r} \int_0^r r'^2 \rho(r') \, dr' + \int_r^\infty r' \rho(r') \, dr' \right].

    - The first term accounts for the mass enclosed within radius :math:`r`.
    - The second term accounts for the contribution from mass at larger radii.

    The importance of the ``boundary_mode`` argument is the nature of the second term. If :math:`\rho(r)` is known over all
    :math:`\mathbb{R}`, then quadrature can be used to evaluate the second term all the way out to :math:`\infty`. If :math:`\rho(r)` is not
    known all the way out (as is the case for splines), then attempting to use standard quadrature will lead to massive inaccuracies. To counteract this,
    the second term is broken up into two terms:

    .. math::

        \int_r^{r_0} r' \rho(r')\, dr' + \int_{r_0}^{\infty} r' \tilde{\rho}(r')\, dr',

    where :math:`\tilde{\rho}` is an **adapted density**. In this case, we let

    .. math::

        \tilde{\rho}(r) = \rho(r_0)\left(\frac{r}{r_0}\right)^\gamma,

    where :math:`\gamma` is the ``powerlaw_index`` parameter. If :math:`\gamma < -2`, then

    .. math::

        \int_{r_0}^{\infty} r' \tilde{\rho}(r') \, dr' = \frac{\rho(r_0)}{r_0^\gamma}\int_{r_0}^{\infty} r'^{\gamma + 1} \, dr' = - \frac{\rho(r_0)r_0^2}{\gamma + 2}.

    This approach then allows for the user to specify density profiles which are not complete on the entire domain but to
    still get accurate results.

    .. rubric:: References

    .. footbibliography::

    See Also
    --------
    solve_poisson_ellipsoidal
        Solve the poisson problem for ellipsoidal density distributions.

    """
    # Ensure density_profile is callable, using interpolation if necessary
    if not callable(density_profile):
        density_profile = InterpolatedUnivariateSpline(coordinates, density_profile)

    # Compute the inner integral (enclosed mass contribution)
    integrand_inner = lambda r: density_profile(r) * r**2
    inner_integral = (1 / coordinates) * integrate_from_zero(
        integrand_inner, coordinates
    )

    # Compute the middle integral (from each radius to the maximum)
    integrand_middle = lambda r: density_profile(r) * r
    middle_integral = integrate(
        integrand_middle, coordinates, x_0=float(coordinates[-1]), minima=False
    )

    # Handle outer boundary using the specified method
    if boundary_mode == "integrate":
        outer_integral = _compute_spherical_poisson_boundary_integrated(
            density_profile, float(coordinates[-1])
        )
    elif boundary_mode == "asymptotic":
        if powerlaw_index is None:
            raise ValueError(
                "`solve_poisson_spherical` cannot compute the boundary value for an array-like density"
                " profile without a specified `powerlaw_index`."
            )
        outer_integral = _compute_spherical_poisson_boundary_asymptotic(
            density_profile, float(coordinates[-1]), powerlaw_index
        )
    else:
        raise ValueError("`boundary_mode` must be 'integrate' or 'asymptotic'.")

    # Combine integrals to compute the gravitational potential
    potential = -4 * np.pi * (inner_integral + middle_integral + outer_integral)

    return potential


def _compute_ellipsoidal_psi_boundary_from_spline(
    density_profile: Callable, r0: float, r_inner: float = 0.0, n: int = -3
) -> float:
    r"""
    Compute the asymptotic boundary contribution for the ellipsoidal Poisson problem
    using a density profile provided as a callable (e.g., a spline).

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\\rho(r)`, which must be callable.
    r0 : float
        The radius beyond which the asymptotic approximation is applied.
    r_inner : float, optional
        The lower limit of integration for the density profile. Default is ``0.0``.
    n : int, optional
        The power-law index :math:`l` for the asymptotic density profile. Default is ``-3``.

    Returns
    -------
    float
        The computed value of :math:`\\psi(\\infty)`.

    Raises
    ------
    ValueError
        If the power-law index ``n`` is greater than or equal to ``-2`` (causing divergence).
    """
    if n >= -2:
        raise ValueError(
            "Boundary behavior (`n`, power index) must be less than -2 for convergence."
        )

    # Integrate the density profile up to r0
    integrand = lambda _xi: _xi * density_profile(_xi)
    psi_integral = 2 * quad(integrand, r_inner, r0)[0]

    # Approximate the contribution beyond r0 using asymptotic behavior
    rho_0 = density_profile(r0)
    psi_inf = psi_integral - (2 * rho_0 * (r0**2)) / (n + 2)

    return psi_inf


def _compute_ellipsoidal_psi_boundary_from_function(density_profile: Callable) -> float:
    r"""
    Compute the complete boundary contribution for the ellipsoidal Poisson problem
    by integrating the density profile over the entire domain.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\\rho(r)`, which must be callable.

    Returns
    -------
    float
        The computed value of :math:`\\psi(\\infty)`.

    Notes
    -----
    This method assumes that the density profile is well-defined for all radii
    (i.e., from :math:`0` to :math:`\infty`) and does not require asymptotic approximations.
    """
    integrand = lambda _xi: _xi * density_profile(_xi)
    return 2 * quad(integrand, 0, np.inf)[0]


def _build_ellipsoidal_psi_abscissa(
    r_min: float, r_max: float, n_points: int, scale: str = "log"
) -> np.ndarray:
    r"""
    Build the abscissa for integration, ensuring consistent scaling and behavior.

    Parameters
    ----------
    r_min : float
        The minimum radius for the integration.
    r_max : float
        The maximum radius for the integration.
    n_points : int
        The number of points for the grid.
    scale : str, optional
        The scale of the grid, either ``log`` or ``linear``. Default is ``log``.

    Returns
    -------
    np.ndarray
        The array of radii for integration.

    Raises
    ------
    ValueError
        If ``scale`` is not ``log`` or ``linear``.
    """
    if r_min < 0:
        raise ValueError("The minimum radius may not be negative.")

    if scale == "log":
        return np.geomspace(r_min, r_max, n_points)
    elif scale == "linear":
        return np.linspace(r_min, r_max, n_points)
    else:
        raise ValueError(f"Unknown scale '{scale}'")


def _compute_ellipsoidal_psi_spline(
    density_profile: Callable,
    r_min: float,
    r_max: float,
    num_points: int,
    n: int,
    scale: str = "log",
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Compute the ellipsoidal :math:`\\psi(r)` using a spline-interpolated density profile.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\\rho(r)`.
    r_min : float
        The minimum radius for the integration.
    r_max : float
        The maximum radius for the integration.
    num_points : int
        Number of points for the integration grid.

    Returns
    -------
    Any
        The radii, the computed :math:`\\psi(r)` values, and the asymptotic limit :math:`\\psi(\\infty)`.

    Notes
    -----
    This method computes \n:math:`\\psi(r)` by numerically integrating the density profile
    up to a maximum radius and approximating the contribution beyond using an asymptotic
    approximation.
    """
    # Build the abscissa
    radii = _build_ellipsoidal_psi_abscissa(r_min, r_max, num_points, scale)

    # Integrate up to each radius
    integrand = lambda _xi: _xi * density_profile(_xi)
    psi_integral = 2 * np.array([quad(integrand, 0, r)[0] for r in radii])

    # Compute the asymptotic limit
    psi_inf = _compute_ellipsoidal_psi_boundary_from_spline(
        density_profile, r_max, r_min, n
    )

    return radii, psi_integral, psi_inf


def _compute_ellipsoidal_psi_function(
    density_profile: Callable, r_min: float, r_max: float, num_points: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Compute the ellipsoidal :math:`\\psi(r)` using a function-defined density profile.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\\rho(r)`, which must be callable.
    r_min : float
        The minimum radius for the integration.
    r_max : float
        The maximum radius for the integration.
    num_points : int
        Number of points for the integration grid.
    **kwargs : dict
        Additional arguments for controlling the behavior, such as ``scale``.

    Returns
    -------
    Any
        The radii, the computed :math:`\\psi(r)` values, and the asymptotic limit :math:`\\psi(\\infty)`.

    Notes
    -----
    This method assumes that the density profile is well-defined for all radii and uses
    direct integration to compute :math:`\\psi(r)`.
    """
    # Build the abscissa
    scale = kwargs.pop("scale", "log")
    radii = _build_ellipsoidal_psi_abscissa(r_min, r_max, num_points, scale)

    # Define the integrand for :math:`\\psi(r)`
    integrand = lambda _xi: _xi * density_profile(_xi)

    # Compute :math:`\\psi(r)` using integration from zero
    psi_integral = 2 * np.array([quad(integrand, 0, r)[0] for r in radii])

    # Add the zero-point for consistency
    psi_integral = np.concatenate([np.array([0]), psi_integral])
    radii = np.concatenate([np.array([0]), radii])

    # Compute the asymptotic limit using the full integral
    psi_inf = _compute_ellipsoidal_psi_boundary_from_function(density_profile)

    return radii, psi_integral, psi_inf


def compute_ellipsoidal_psi(
    density_profile: Callable,
    r_min: float,
    r_max: float,
    num_points: int = 1000,
    method: str = "spline",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Compute the ellipsoidal :math:`\psi(r)` function using a spline interpolation scheme.

    This function takes a function :math:`\rho(r)` representing the dynamical density at effective radius :math:`r` and
    returns

    .. math::

        \psi(r) = 2\int_0^r \; d\xi\; \xi \rho(\xi),

    at each :math:`r` specified over a particular domain. :py:func:`compute_ellipsoidal_psi` is intended to be a general purpose
    solver for this purpose and can manage cases when ``density_profile`` is fully supported on :math:`\mathbb{R}` and also
    in cases where it is not.

    The :math:`\psi(r)` function is utilized in quadrature solutions of the `Poisson equation <https://en.wikipedia.org/wiki/Poisson%27s_equation>`_
    in cases of ellipsoidal symmetry.

    Parameters
    ----------
    density_profile : Callable
        The density profile function :math:`\rho(r)`, which must be callable.

        This should either be a spline approximation of the density profile, or a callable completely characterizing
        the density profile.

        .. note::

            If you only have a spline for ``density_profile``, then its behavior above and below the domain boundaries
            is not well constrained. In such a case, ``r_min``, ``r_max``, and ``method`` should be specified correctly
            to ensure that a reasonable boundary approximation is used.

    r_min : float
        The minimum radius for the integration. This should be sufficiently close to zero that :math:`\rho(r)` can be
        linearly estimated between zero and ``r_min``.

        .. warning::

            ``r_min`` must be greater than 0.

    r_max : float
        The maximum radius for the integration. In general, this should simply be the largest effective radius for
        which the :math:`\psi(r)` function is necessary.
    num_points : int
        Number of points for the integration grid. The output abscissa will be ``num_points + 1`` in size to include
        ``0``.
    method: str
        The method for computing the ellipsoidal :math:`\psi(r)`.

        - ``"spline"``: Assumes that ``density_profile`` was provided as a spline and an asymptotic approximation is
          used for the extrapolation to :math:`\infty`.
        - ``"function"``: Assumes that the ``density_profile`` was provided as a function that it's asymptotic behavior
          is completely characterized. In this case, the boundary limit is determined by quadrature to :math:`\infty`.

        If possible, ``"function"`` is the more accurate option; however, if the ``density_profile`` is not fully
        supported to :math:`\infty`, then ``"function"`` will yield massively erroneous results for the boundary.

    **kwargs
        Additional arguments passed to sub-function solvers.

        - If ``method = 'spline'``, then

          - ``scale``: may be either ``"log"`` or ``"linear"``, determines the spacing between points in the
            interpolation abscissa.
          - ``n``: the power-law index at large radii estimating :math:`\rho(r)`. This is used to obtain the
            boundary value.

        - If ``method = 'function'``, then

          - ``scale``: may be either ``"log"`` or ``"linear"``, determines the spacing between points in the
            interpolation abscissa.


    Returns
    -------
    np.ndarray
        The abscissa of the computed :math:`\psi(r)` values. This includes :math:`0`, but not :math:`\infty`. This
        array will be of shape ``(num_points + 1, )``.
    np.ndarray
        The computed values of :math:`\psi(r)` over the domain. This array will be of shape ``(num_points + 1, )``.
    float
        The asymptotically computed limit :math:`\lim_{r\to \infty} \psi(r)`.

    Raises
    ------
    ValueError
        If an invalid ``method`` is provided.

    Notes
    -----
    This function combines spline and function-based methods to compute :math:`\psi(r)` depending on the domain
    and characteristics of the density profile.
    """
    if method == "spline":
        return _compute_ellipsoidal_psi_spline(
            density_profile, r_min, r_max, num_points, **kwargs
        )
    elif method == "function":
        return _compute_ellipsoidal_psi_function(
            density_profile, r_min, r_max, num_points, **kwargs
        )
    else:
        raise ValueError("`method` must be 'spline' or 'function'.")


def solve_poisson_ellipsoidal(
    density_profile: Union[Callable, np.ndarray],
    coordinates: np.ndarray,
    coordinate_system: "PseudoSphericalCoordinateSystem",
    /,
    num_points: int = 1000,
    *,
    scale: str = "log",
    psi: Callable = None,
    powerlaw_index: int = None,
) -> np.ndarray:
    r"""
    Solve Poisson's equation for a system with iso-density curves in similar ellipsoids.

    This function computes the gravitational potential :math:`\Phi(\mathbf{x})` for a density profile
    :math:`\rho(r)` in ellipsoidal coordinates, where the effective radius is defined as:

    .. math::

        m^2 = \sum_i \eta_i^2 x_i^2,

    and :math:`\eta_i` are the scale parameters of the ellipsoidal coordinate system
    (see :py:class:`~pisces.geometry.coordinate_systems.PseudoSphericalCoordinateSystem`).

    Parameters
    ----------
    density_profile : Callable or np.ndarray
        The density profile function :math:`\rho(r)`.
        Can be:

        - A callable function defining the density.
        - A numerical array corresponding to density values at specified radii. If provided, it will
          be interpolated to create a callable function.

    coordinates : np.ndarray
        Array of ellipsoidal coordinates where the potential is computed. Must follow the format
        ``(..., NDIM)``, where ``NDIM`` is the number of dimensions.
    coordinate_system : PseudoSphericalCoordinateSystem
        The coordinate system defining the ellipsoidal geometry, including scale parameters.
    num_points : int, optional
        Number of points for integration grids. Default is ``1000``.
    scale : str, optional
        Scaling method for grid construction. Can be ``'log'`` or ``'linear'``. Default is ``'log'``.
    psi : Callable, optional
        Precomputed :math:`\psi(r)` function for the density profile. If provided, this will be
        used directly. If ``None``, it will be computed from the density profile.

        .. note::

            :math:`\psi(r)` is defined as

            .. math::

                \psi(r) = 2\int_0^r \; \xi \rho(\xi) \; d\xi.

    powerlaw_index : int, optional
        Power-law index for the asymptotic density behavior. Required when ``density_profile`` is
        an array and ``psi`` is not provided.

    Returns
    -------
    np.ndarray
        Array of gravitational potential values at the specified coordinates.

    Raises
    ------
    ValueError
        If the input density profile is not callable and ``powerlaw_index`` is not specified.
    ValueError
        If the coordinate system's scale parameters cannot be extracted.

    Notes
    -----
    The potential is computed as :footcite:p:`BovyGalaxyBook`:

    .. math::
        \Phi(\mathbf{x}) = -\pi \frac{1}{(\prod_i \eta_i)^2} \int_0^\infty
        \frac{\psi(\infty) - \psi(\xi(\tau))}{\sqrt{\prod_i (\tau + \eta_i^{-2})}} \, d\tau,

    where:

    - :math:`\psi(r)` is computed based on the density profile.
    - :math:`\xi(\tau)` is the effective radius for a given integration parameter :math:`\tau`.

    For more detail on the relevant theory, see :ref:`poisson_equation`.

    .. rubric:: References

    .. footbibliography::

    See Also
    --------
    solve_poisson_spherical


    """
    # Manipulate the coordinates to ensure they need our formatting constraints and expectations.
    #   - coordinates should be reformatted to our (...,NDIM) formatting standard.
    #   - coordinates are converted to cartesian coordinates for the integration procedure.
    #   - The minimum and maximum radii are extracted.
    coordinates = CoordinateArray(coordinates, coordinate_system.NDIM)
    cartesian_coordinates = coordinate_system.to_cartesian(coordinates)
    r_min, r_max = np.amin(coordinates[..., 0]), np.amax(coordinates[..., 0])
    # Extract the scale parameters and generate the necessary scale arrays for the
    # procedures that need to be carried out.
    try:
        sx, sy, sz = (
            coordinate_system.scale_x,
            coordinate_system.scale_y,
            coordinate_system.scale_z,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to extract scale parameters from input coordinate system ({coordinate_system}): {e}."
        )

    _scale_array_shape = [1] * (
        coordinates.ndim - 1
    )  # Need N-1 slots to make broadcastable.
    _unit_scale_array_base = np.ones(_scale_array_shape)

    # produce the scale product and the inverse square array (both used later).
    scale_product = sx * sy * sz
    inverse_square_array = np.stack(
        [
            _scale_parameter ** (-2) * _unit_scale_array_base
            for _scale_parameter in [sx, sy, sz]
        ],
        axis=-1,
    )

    # @@ CONSTRUCT PSI @@ #
    # This is the most complex procedure in this function. Depending on how we are given
    # the density profile, we need to proceed with different methodologies and require different
    # values for the inputs.
    if psi is None:
        # Validation step: ensure that we have everything we need to proceed.
        if (not callable(density_profile)) and (powerlaw_index is None):
            raise ValueError(
                "`solve_poisson_ellipsoidal` cannot compute the psi function from an array-like density"
                " profile without a specified `powerlaw_index`."
            )

        # Spline generation step. If we don't already have a spline, we need to build one.
        # this will now make things callable.
        if not callable(density_profile):
            _psi_method = "spline"
            radii = _build_ellipsoidal_psi_abscissa(r_min, r_max, num_points, scale)
            density_profile = InterpolatedUnivariateSpline(radii, density_profile)
            _kwgs = dict(scale=scale, n=powerlaw_index)
        else:
            _psi_method = "function"
            _kwgs = dict(scale=scale)

        # Compute the psi function abscissa and interpolation values.
        radii, psi_values, psi_inf = compute_ellipsoidal_psi(
            density_profile,
            r_min,
            r_max,
            num_points=num_points,
            method=_psi_method,
            **_kwgs,
        )

        # Create the necessary spline
        psi_func = InterpolatedUnivariateSpline(radii, psi_values)
    else:
        # We already have psi specified so we can just run with that.
        psi_func = psi
        psi_inf = psi_func(np.inf)

    # @@ CONSTRUCT INTEGRANDS @@ #
    # At this stage, we construct the numerator and denominator functions so
    # that we can proceed.
    xi_func = lambda _tau: np.sqrt(
        np.sum(cartesian_coordinates**2 / (_tau + inverse_square_array), axis=-1)
    )
    denom_func = lambda _tau: np.sqrt(np.prod(_tau + inverse_square_array, axis=-1))

    # Define the integrand for the potential
    integrand = lambda _tau: (psi_inf - psi_func(xi_func(_tau))) / denom_func(_tau)

    # Compute the integral using vectorized quadrature
    potential = -np.pi * (1 / scale_product**2) * quad_vec(integrand, 0, np.inf)[0]

    return potential
