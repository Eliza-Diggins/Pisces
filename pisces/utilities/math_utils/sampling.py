"""
Sampling utilities for Pisces.
"""
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid

from pisces.utilities.config import pisces_params
from pisces.utilities.math_utils._sampling_opt import (
    rejection_sampling_2D,
    rejection_sampling_2D_proposal,
    rejection_sampling_3D,
    rejection_sampling_3D_proposal,
)


# noinspection PyTypeChecker
def rejection_sample(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    proposal: Optional[Union[Callable, np.ndarray]] = None,
    proposal_axis: int = 0,
    chunk_size: int = None,
    max_iter: int = 10_000,
    out: Optional[Any] = None,
):
    r"""
    Perform rejection sampling in 2D or 3D space to generate samples from a function proportional to a
    particular probability density function.

    Parameters
    ----------
    x : np.ndarray
        The coordinate grid where the PDF ``y`` is defined. This array must have shape ``(N1, N2, ..., N_ndim, ndim)``
        where ``ndim`` is the number of dimensions (2 or 3). The last axis represents the coordinate axes
        of the grid (e.g., ``[x, y]`` or ``[x, y, z]``). All but the last axis must match the shape of ``y``.

        .. hint::

            This follows the standard convention of coordinate grids in Pisces. If you have a coordinate grid already,
            it should be valid for this function.

    y : np.ndarray
        The likelihood function to sample from. The likelihood function must be proportional to the underlying PDF, but
        does not need to be normalized. It should be provided at each point on the coordinate grid. This array must
        have shape ``(N1, N2, ..., N_ndim)``, where each element corresponds to a point on the grid ``x``.

    n_samples : int
        The number of samples to generate.

    proposal : Optional[Union[Callable, np.ndarray]], optional
        The proposal distribution for rejection sampling. If ``None``, the algorithm uses a uniform
        proposal distribution. The proposal should be a 1-D callable or array specifying the values of the likelihood at
        each point along a particular axis of the abscissa (``x``). The ``proposal_axis`` argument dictates which axis of
        the domain is used as the support for the proposal. If the proposal is an array, it must be the same shape as the
        corresponding axis of the abscissa.

    proposal_axis : int, optional
        The axis along which the proposal distribution is applied. Default is ``0``.

    chunk_size : int, optional
        The size of chunks to generate in each iteration of rejection sampling. Default is ``None``,
        which sets it to the maximum of 1e6 or ``n_samples``.

    max_iter : int, optional
        The maximum number of iterations allowed to generate the required number of samples. Default is ``10,000``.

    out : Optional[Any], optional
        An optional pre-allocated array to store the output samples. If ``None``, a new array of shape
        ``(n_samples, ndim)`` is created.

    Returns
    -------
    np.ndarray
        An array of shape ``(n_samples, ndim)`` containing the generated samples.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have incompatible shapes or dimensions.
        If ``ndim(x) < 2`` or ``ndim(x) > 3`` (unsupported dimensionality).
        If ``proposal`` is provided and has an invalid shape or is incompatible with the other inputs.

    Notes
    -----
    This method supports rejection sampling in 2D and 3D only. Higher-dimensional spaces are not
    currently supported. For lower-dimensional cases, consider using inverse transform sampling.

    If a proposal distribution is provided, it is evaluated (if callable) or used directly to guide
    the sampling process. The algorithm ensures the output samples follow the target PDF `y` as closely
    as possible, using the proposal to improve efficiency when appropriate.

    Examples
    --------

    As an example, let's draw samples from the 2-D gaussian distribution defined by

    .. math::

        \mathcal{L}(x,y) = e^{-(2x^2+y^2)}

    on the domain :math:`[-3,3]\times[-3,3] \subset \mathbb{R}^2`. To do so, we first need to construct the abscissa and
    the values of the field:

    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.utilities.math_utils.sampling import rejection_sample
        >>>
        >>> xmin,xmax = -2,2
        >>> x,y = np.linspace(xmin,xmax,1000),np.linspace(xmin,xmax,1000)
        >>> x_hist,y_hist = np.linspace(xmin,xmax,100),np.linspace(xmin,xmax,100)
        >>> X,Y = np.meshgrid(x,y,indexing='ij') # The indexing is CRITICAL here.
        >>> abscissa = np.moveaxis(np.stack([X,Y],axis=0),0,-1)
        >>> field = np.exp(-1*(2*abscissa[...,0]**2 + abscissa[...,1]**2))
        >>>
        >>> # Proceed to sample from the field
        >>> samples = rejection_sample(abscissa,field,1000000,chunk_size=10000)
        >>>
        >>> # Produce a histogram of the samples
        >>> hist_image,_,_ = np.histogram2d(samples[:,0],samples[:,1],bins=(x_hist,y_hist))
        >>> # Generate the plot.
        >>> fig,axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,5),gridspec_kw={'wspace':0.1})
        >>> _ = axes[0].imshow(hist_image.T/np.amax(hist_image.T), origin='lower', extent=[xmin,xmax,xmin,xmax],cmap='inferno')
        >>> _ = axes[1].imshow(field.T/np.amax(field.T),origin='lower',extent=[xmin,xmax, xmin,xmax],cmap='inferno')
        >>> _ = plt.colorbar(plt.cm.ScalarMappable(cmap='inferno'),ax=axes,fraction=0.07)
        >>> _ = axes[0].set_ylabel('y')
        >>> _ = axes[1].set_xlabel('x')
        >>> _ = axes[0].set_xlabel('x')
        >>> _ = axes[1].set_title("Likelihood Function")
        >>> _ = axes[0].set_title("Sampled Values")
        >>> plt.show()

    """
    # Validate user provided arguments. Ensure that x and y are commensurate and
    # that we can perform the sampling procedure on it.
    _x_shape = x.shape
    _y_shape = y.shape

    # Check that x and y are valid shapes.
    ndim = x.ndim - 1
    if not np.array_equal(_x_shape[:-1], _y_shape):
        raise ValueError(
            f"rejection_sample got x argument with shape {_x_shape} and y argument with shape {_y_shape}.\n"
            f"Function expects N-1 commensurate axes, which is not the case here."
        )

    # check that the last axis has the correct dimensions and that y has the correct dimensions.
    if y.ndim != ndim:
        raise ValueError(
            f"rejection_sample detected that x was {ndim} dimensional coordinate grid, but y was {y.ndim} dimensional.\n"
            f"Function expects ndim(x) = ndim(y)."
        )

    if _x_shape[-1] != ndim:
        raise ValueError(
            f"Coordinate grid provide as `x` argument was {ndim} dimensional. "
            f"Must have {ndim} coordinate axes, not {_x_shape[-1]}."
        )

    # Check that the dimension is either 2 or 3. We don't support higher and lower should be done
    # with inverse sampling.
    if ndim < 2:
        raise ValueError(
            f"Determined `x` to be {ndim}, which should be performed with inverse transform sampling."
        )
    elif ndim > 3:
        raise ValueError(
            f"Determined `x` to be {ndim}, but only 2 and 3 dimensional rejection sampling are supported."
        )

    # Manage the arguments that need to be filled if they aren't provided.
    if chunk_size is None:
        chunk_size = np.amax([1e6, n_samples])  # Cap at 1e6 to be generally beneficial.

    if out is None:
        # create an out array from scratch.
        out = np.empty((n_samples, ndim), dtype=np.float64)

    # Manage the proposal. If we got a proposal, we need to determine if it's a callable and
    # then evaluate it on the abscissa.
    if proposal is not None:
        # Construct an axis slice so that we can extract the evaluation domain for the
        # proposal more elegantly that we were forced to do in the Cython.
        _prop_axis_slice = list([0] * ndim + [proposal_axis])
        _prop_axis_slice[proposal_axis] = slice(None)
        _prop_axis_slice = tuple(_prop_axis_slice)

        # Check if the proposal is callable. If it is we need to evaluate on the correct
        # portion of the abscissa.
        if callable(proposal):
            x_proposal = x[_prop_axis_slice]
            proposal = proposal(x_proposal)

        # Pass the data on to the low-level callable
        if ndim == 2:
            rejection_sampling_2D_proposal(
                x,
                y,
                n_samples,
                chunk_size,
                proposal,
                proposal_axis,
                out,
                max_iter,
                int(pisces_params["system.preferences.disable_progress_bars"]),
            )
        elif ndim == 3:
            rejection_sampling_3D_proposal(
                x,
                y,
                n_samples,
                chunk_size,
                proposal,
                proposal_axis,
                out,
                max_iter,
                int(pisces_params["system.preferences.disable_progress_bars"]),
            )
    else:
        if ndim == 2:
            rejection_sampling_2D(
                x,
                y,
                n_samples,
                chunk_size,
                out,
                max_iter,
                int(pisces_params["system.preferences.disable_progress_bars"]),
            )
        elif ndim == 3:
            rejection_sampling_3D(
                x,
                y,
                n_samples,
                chunk_size,
                out,
                max_iter,
                int(pisces_params["system.preferences.disable_progress_bars"]),
            )

    # Finish and return the output array
    return out


def inverse_transform_sample(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    out: Optional[np.ndarray] = None,
    integrator: Literal["simpson", "trapezoid"] = "trapezoid",
) -> np.ndarray:
    r"""
    Perform inverse transform sampling for a 1D probability density function (PDF).

    Parameters
    ----------
    x : np.ndarray
        The abscissa values corresponding to the PDF values in ``y``. Must be monotonically increasing.

    y : np.ndarray
        The probability density function values at the abscissa points in ``x``. Does not need to be normalized,
        but must be non-negative.

    n_samples : int
        The number of samples to generate.

    out : Optional[np.ndarray], optional
        An optional pre-allocated array to store the output samples. If ``None``, a new array of shape ``(n_samples,)``
        is created.
    integrator: str, optional
        The integrator to use to construct the cumulative. This may be either ``"simpson"`` or ``"trapezoid"``.

    Returns
    -------
    np.ndarray
        An array of shape ``(n_samples,)`` containing the generated samples.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have incompatible shapes, or if ``y`` contains negative values.

    Notes
    -----
    This method constructs the cumulative distribution function (CDF) via trapezoidal summation, normalizes the CDF,
    and uses 1D linear interpolation to map uniformly distributed random numbers to samples from the target PDF.

    Examples
    --------
    Example: Sampling from a Gaussian PDF

    .. plot::
        :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pisces.utilities.math_utils.sampling import inverse_transform_sample
        >>>
        >>> x = np.linspace(-5, 5, 1000)
        >>> pdf = np.exp(-0.5 * x**2)  # Gaussian PDF (unnormalized)
        >>> samples = inverse_transform_sample(x, pdf, 10000)
        >>>
        >>> _ = plt.hist(samples, bins=50, density=True, alpha=0.6, label="Samples")
        >>> _ = plt.plot(x, pdf / np.trapz(pdf, x), label="PDF", color="red")
        >>> _ = plt.legend()
        >>> plt.show()

    """
    # Input validation. Check shapes and ensure that all of the likelihood function values are
    # larger than 0. Construct the output array if not provided.
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, but got {x.shape} and {y.shape}."
        )
    if np.any(y < 0):
        raise ValueError("y must be non-negative.")
    if out is None:
        out = np.empty(n_samples, dtype=np.float64)

    if integrator == "trapezoid":
        _integrator = cumulative_trapezoid
    elif integrator == "simpson":
        _integrator = cumulative_simpson
    else:
        raise ValueError(f"The integrator {integrator} is not supported.")

    # Construct the cumulative distribution using either simpson quadrature or the
    # trapazoidal sum over the elements.
    cdf = _integrator(y, x, initial=0)
    cdf /= cdf[-1]  # Normalize the CDF to 1

    # Generate uniform random numbers in [0, 1]
    # Interpolate the uniform random numbers onto the inverse CDF to generate samples
    u = np.random.uniform(0, 1, size=n_samples)
    out[:] = np.interp(u, cdf, x)

    return out


def random_sample_spherical_angles(num_points: int):
    r"""
    Generate random samples of spherical angles (theta and phi).

    This function samples `num_points` random points on a sphere by generating
    uniformly distributed angles in spherical coordinates. The angles are:
    - `theta` (polar angle): Distributed such that the points are uniformly spread on the sphere.
    - `phi` (azimuthal angle): Uniformly distributed in the range [0, 2π).

    Parameters
    ----------
    num_points : int
        The number of random points to generate.

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - `theta` : np.ndarray
            The polar angles in radians, uniformly sampled for spherical symmetry.
        - `phi` : np.ndarray
            The azimuthal angles in radians, uniformly sampled in the range [0, 2π).

    Raises
    ------
    ValueError
        If `num_points` is not a positive integer.

    Notes
    -----
    - `theta` is derived from `cos(theta)` values uniformly distributed in [-1, 1],
      ensuring uniform coverage of the sphere's surface.
    - `phi` is sampled uniformly in the range [0, 2π).
    """
    if num_points <= 0:
        raise ValueError("`num_points` must be a positive integer.")

    # Uniform sampling of cos(theta) for uniform spherical distribution
    theta = np.arccos(np.random.uniform(-1, 1, size=num_points))

    # Uniform sampling of phi in [0, 2π)
    phi = 2 * np.pi * np.random.uniform(0, 1, size=num_points)

    return theta, phi
