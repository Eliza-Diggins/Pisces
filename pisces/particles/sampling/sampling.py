"""
Sampling utilities for drawing particles from density distributions.
"""
from typing import Any, Callable, Dict, Tuple

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pisces.particles.sampling._invsamp import invcsamp_cdf
from pisces.particles.sampling._rejsamp import (
    brejs1d,
    brejs1d_hdf5,
    brejs2d,
    brejs2d_hdf5,
    brejs3d,
    brejs3d_hdf5,
    prejs1d,
    prejs1d_hdf5,
    prejs2d,
    prejs2d_hdf5,
    prejs3d,
    prejs3d_hdf5,
)
from pisces.utilities.array_utils import CoordinateGrid, extend_and_pad
from pisces.utilities.logging import mylog
from pisces.utilities.math_utils.numeric import create_cdf

# @@ STATIC VARIABLES @@ #
_rejection_sampler_map: Dict[Tuple[int, bool, bool], Callable] = {
    (1, False, False): brejs1d,
    (2, False, False): brejs2d,
    (3, False, False): brejs3d,
    (1, False, True): brejs1d_hdf5,
    (2, False, True): brejs2d_hdf5,
    (3, False, True): brejs3d_hdf5,
    (1, True, False): prejs1d,
    (2, True, False): prejs2d,
    (3, True, False): prejs3d,
    (1, True, True): prejs1d_hdf5,
    (2, True, True): prejs2d_hdf5,
    (3, True, True): prejs3d_hdf5,
}


# @@ SAMPLING METHODS @@ #
# These are the standard samplers implemented in Pisces.
def sample_inverse_cumulative(
    x: ArrayLike, y: ArrayLike, n_samples: int, /, bounds: ArrayLike = None
) -> NDArray[float]:
    r"""
    Draw samples for a pdf ``y`` by computing the cumulative distribution and sampling
    from the inverse CDF.

    Parameters
    ----------
    x: ArrayLike
        The abscissa (domain) of the pdf function. ``x`` should be a 1-D array-like object.
    y: ArrayLike
        The ordinates of the pdf function. Must match the shape of ``x``.
    n_samples: int
        The number of samples to draw.
    bounds: ArrayLike, optional
        The bounds on the domain. This may be a length 2 array-like object specifying the left and
        right bounds of the domain. If either bound is within the existing abscissa specified by ``x``, an
        error is raised. The PDF function ``y`` is simply padded out to the boundaries.

    Returns
    -------
    np.ndarray
        The resulting samples from the distribution. This will be an ``n_samples`` length array.

    Examples
    --------
    As an example, one can easily draw samples from a normal distribution using inverse cumulative sampling. For
    this example, we'll consider the un-normalized pdf

    .. math::

        f(x) = \exp\left(-x^2\right)

    If we sample a set of particles from this likelihood, it should be distributed according to the
    generic pdf of a normal variate.

    .. plot::

        >>> from pisces.particles.sampling import sample_inverse_cumulative
        >>> from scipy.stats import norm
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # Create the abscissa and likelihood for our custom likelihood.
        >>> x_grid = np.linspace(-5,5,100)
        >>> y_grid = np.exp(-(x_grid**2)/2)
        >>>
        >>> # Draw samples from the distribution.
        >>> samples = sample_inverse_cumulative(x_grid,y_grid,1_000_000,bounds=(-6,6))
        >>>
        >>> # plot the histogram and the true distribution.
        >>> hist, edges = np.histogram(samples, bins=100, density=True)
        >>> bin_centers = (edges[1:]+edges[:-1])/2
        >>> _ = plt.plot(bin_centers, hist)
        >>> _ = plt.plot(bin_centers, norm.pdf(bin_centers))
        >>> plt.show()


    """
    # Construct the cumulative distribution function by passing to the
    # math utilities level. This may extend the abscissa, which means we want to
    # keep the extended x values.
    x_extended, cdf = create_cdf(x, y, bounds=bounds)

    # Pass to the cython level and invoke invcsamp_cdf, which will then
    # actually perform the sampling and allow us to return the result.
    # ! inspection is suppressed because the cython inspection is not good.
    # noinspection PyTypeChecker
    return invcsamp_cdf(
        np.asarray(x_extended, dtype="f8"), np.asarray(cdf, dtype="f8"), n_samples
    )


def rejection_sample(
    x: ArrayLike,
    y: ArrayLike,
    n_samples: int,
    /,
    result_buffer=None,
    bounds: ArrayLike = None,
    proposal: ArrayLike = None,
    paxis: int = None,
    chunk_size: int = 100_000,
    max_iterations: int = 10_000,
    show_progress: bool = True,
) -> Any:
    r"""
    Perform a rejection-acceptance sampling scheme in 1-3 dimensions using either
    a uniform proposal or a 1D proposal distribution. This function supports writting
    to HDF5 style buffers.

    Parameters
    ----------
    x: ArrayLike
        An array like object representing the abscissa over which the likelihood is interpolated. This
        should be a valid coordinate grid in shape (i.e. ``(Nx,Ny,Nz,3)``) and must match the shape of ``y``
        up to the final axis. Each of the first axes represents the grid index along that coordinate direction while
        the final axis represents the actual coordinate values.
    y: ArrayLike
        An array containing the value of the likelihood at each point in the coordinate grid provided by ``x``.
    n_samples: int
        The number of samples to draw.
    bounds: ArrayLike, optional
        Bounds to place on the sampling domain. This must be a ``2*ndim`` array of minimum and maximum values along each
        axis.

        .. hint::

            This is generally useful when the abscissa (``x``) doesn't go all the way to the boundary of the domain
            so you want to ensure that all points in the desired domain are sampleable.

        If bounds fall within the provided abscissa, then an error is raised.

    proposal: ArrayLike, optional
        The proposal distribution to use for sampling along a single axis. This array must be the sample size
        as the corresponding axis of the grid and should contain the proposal likelihood at each point on that
        axis.
    paxis: int, optional
        The axis along which the proposal distribution is provided. This must be specified if ``proposal`` is specified.
    result_buffer: ArrayLike, optional
        The result buffer into which the samples should be placed. If ``result_buffer = None`` (default), then a new
        numpy array is created to store the samples. If ``result_buffer`` is a numpy array, then it is filled with the
        values from the sample. Finally, an HDF5 dataset may be passed as the result buffer and the hdf5 dataset will
        then be used to write the samples to disk directly.

        .. note::

            If the result buffer is not the correct shape, an error is raised.

    chunk_size: int, optional
        The size of a single chunk of proposals. For each cycle of the algorithm, a single chunk worth of
        proposals are considered and each is either accepted or rejected. By specifying a small chunk size, the
        memory use is reduced, but at the cost of higher computational cost. Likewise, a large chunk will generally
        improve execution time at the expense of memory usage.
    max_iterations: int, optional
        The maximum number of chunk-sized proposals to take. This should be used to set an upper limit on the
        time before which this function will not raise an error.
    show_progress: bool, optional
        If ``True``, then a progress bar will be shown during sampling.

    Returns
    -------
    result_buffer

    Examples
    --------
    **Sampling in 1 Dimension:**

    .. hint::

        It is generally bad practice to use A/R sampling for 1D distributions as inversion sampling
        would be faster.

    In this example, we'll consider sampling from a Gaussian likelihood using naive rejection sampling.

    .. plot::

        >>> from scipy.stats import norm
        >>> from pisces.particles.sampling.sampling import rejection_sample
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>>
        >>> # Create the abscissa and likelihood for our custom likelihood.
        >>> x_grid = np.linspace(-5,5,100)
        >>> y_grid = np.exp(-(x_grid**2)/2)
        >>>
        >>> # Draw samples from the distribution.
        >>> samples = rejection_sample(x_grid,y_grid,1_000_000)
        >>>
        >>> # plot the histogram and the true distribution.
        >>> hist, edges = np.histogram(samples, bins=100, density=True)
        >>> bin_centers = (edges[1:]+edges[:-1])/2
        >>> _ = plt.plot(bin_centers, hist)
        >>> _ = plt.plot(bin_centers, norm.pdf(bin_centers))
        >>> plt.show()

    We have an acceptance rate of approximately 21% for this case. Given that we have a sampling space
    of 12 and the acceptance region takes up

    .. math::

        \int_{-\infty}^{\infty} e^{-x^2/2} dx = \sqrt{2\pi},

    This is precisely the predicted rate. We can improve the rate by selecting a 1D distribution which is easy
    to sample using inversion sampling. In this case, we consider

    .. math::

        p(x) = 1 - \left|x\right|/3

    .. plot::

        >>> from scipy.stats import norm
        >>> import matplotlib.pyplot as plt
        >>> from pisces.particles.sampling.sampling import rejection_sample
        >>> import numpy as np
        >>>
        >>> # Create the abscissa and likelihood for our custom likelihood.
        >>> x_grid = np.linspace(-6,6,100)
        >>> y_grid = np.exp(-(x_grid**2)/2)
        >>> proposal = -np.abs(x_grid)/3 + 1
        >>> # Ensure that the proposal is nowhere <= zero.
        >>> proposal = np.amax([y_grid,proposal],axis=0)
        >>>
        >>> # Draw samples from the distribution.
        >>> samples = rejection_sample(x_grid,y_grid,1_000_000,proposal=proposal,paxis=0)
        >>>
        >>> # plot the histogram and the true distribution.
        >>> hist, edges = np.histogram(samples, bins=100, density=True)
        >>> bin_centers = (edges[1:]+edges[:-1])/2
        >>> _ = plt.plot(bin_centers, hist)
        >>> _ = plt.plot(bin_centers, norm.pdf(bin_centers))
        >>> plt.show()

    This dramatically improves the acceptance rate.

    **Sampling in 2 Dimension:**

    In 2 dimensions, the procedure is very similar:

    .. plot::

        >>> from scipy.stats import norm,beta
        >>> import matplotlib.pyplot as plt
        >>> from pisces.particles.sampling.sampling import rejection_sample
        >>> import numpy as np
        >>>
        >>> # Generate the abscissa
        >>> x,y = np.linspace(-5,5,500),np.linspace(0,1,500)
        >>> X,Y = np.meshgrid(x,y,indexing='ij')
        >>> C = np.moveaxis(np.asarray([X,Y]),0,-1)
        >>>
        >>> # Construct the likelihood function.
        >>> px, py = lambda x: 0.5*(norm(loc=-3).pdf(x)+norm(loc=3).pdf(x)), beta(a=2,b=3).pdf
        >>> Z = px(X)*py(Y)
        >>>
        >>> # Draw samples from the distribution.
        >>> samples = rejection_sample(C,Z,100_000_000)
        >>>
        >>> # Create the histogram of the samples
        >>> hist,ex,ey = np.histogram2d(*samples.T, bins=100, density=True)
        >>> ecx,ecy = (ex[1:]+ex[:-1])/2,(ey[1:]+ey[:-1])/2
        >>>
        >>> # Create the marginal PDFs
        >>> Ix = np.sum(hist*np.diff(ey),axis=1)
        >>> Iy = np.sum(hist*np.diff(ex),axis=0)
        >>>
        >>> # Plot
        >>> fig,axes = plt.subplots(2,2,gridspec_kw=dict(hspace=0,wspace=0,height_ratios=[1,3],width_ratios=[3,1]))
        >>> _ = axes[0,1].set_visible(False)
        >>> _ = axes[1,0].imshow(hist.T,extent=(-5,5,0,1),origin='lower',aspect='auto')
        >>> _ = axes[0,0].plot(ecx,Ix,color='red',ls='-')
        >>> _ = axes[1,1].plot(Iy,ecy,color='blue',ls='-')
        >>> _ = axes[0,0].plot(ecx,px(ecx),color='red',ls=':')
        >>> _ = axes[1,1].plot(py(ecy),ecy,color='blue',ls=':')
        >>> plt.show()
    """
    # Validate the input arrays and the shapes. We need to ensure that
    # everything is correctly shaped and is C-contiguous.
    x, y = np.asarray(x, dtype="f8", order="C"), np.asarray(y, dtype="f8", order="C")
    _number_of_dims = max(x.ndim - 1, 1)

    # Coerce coordinate shape. We require the standard coordinate grid
    # format for this operation even at the Cython / C level.
    x = CoordinateGrid(x, ndim=_number_of_dims)

    # Perform the validation tasks to ensure that all the inputs are
    # valid / legitimate.
    if np.not_equal(x.shape[:-1], y.shape).any():
        # The x and y shape are not consistent.
        raise ValueError(
            f"Coordinates `x` have shape {x.shape} while ordinates `y` have shape {y.shape}.\n"
            "`y` must match the shape of `x` up to the final dimension."
        )
    if x.shape[-1] != _number_of_dims:
        raise ValueError(
            f"Coordinates `x` have shape {x.shape}, which implies {_number_of_dims} dimensions, but"
            f" `x` has only {x.shape[-1]} coordinates specified."
        )
    if proposal is not None:
        if paxis is None:
            raise ValueError(
                "`proposal` must specify `paxis` if `proposal` is not None."
            )
        if len(proposal) != x.shape[paxis]:
            raise ValueError(
                f"`proposal` has length {len(proposal)}, but x has length {x.shape[paxis]} along the paxis. These"
                " must match."
            )

    # Fix the chunk size
    chunk_size = min(chunk_size, n_samples)

    # @@ BOUNDARY MANAGEMENT @@ #
    # In many cases, boundaries are passed which then need to be added
    # to the edges of the abscissa and the ordinates. This is managed
    # in this section of the code.
    if bounds is not None:
        x, y, _lr_extension_array = extend_and_pad(x, y, bounds)

        if proposal is not None:
            # We need to also pad out the proposal to ensure that it is
            # the right size.
            proposal = np.pad(
                proposal,
                pad_width=_lr_extension_array[paxis, :].astype(int),
                mode="edge",
            )

    # @@ REGULARIZATION AND PROPOSAL MANAGEMENT @@ #
    # In this phase, we need to regularize the likelihood so that it maxes out at 1
    # and we also need to alter the proposal so that the likelihood ratio bound is 1.
    y /= np.amax(y)  # Regularizes the likelihood ordinates.

    if proposal is not None:
        # Because there is a proposal, we need to regularize the proposal likelihood,
        # construct the proposal CDF, and construct the proposal abscissa.
        proposal_x = x[
            tuple(0 if _i != paxis else slice(None) for _i in range(_number_of_dims))
        ][:, paxis]

        # Construct the proposal cdf. We don't pass bounds to create_cdf because
        # bounds have already been added to the abscissa.
        _, proposal_cdf = create_cdf(proposal_x, proposal)

        # Regularize the likelihood ratio. Because we need f(x) < Mg(x), we let
        # the proposal become Mg(x) and then we can just compare f(x) < g(x) at the
        # C level without worrying about the bounds.
        _expanded_proposal = proposal.reshape(
            tuple(1 if i != paxis else x.shape[i] for i in range(_number_of_dims))
        )
        likelihood_ratio = y / _expanded_proposal
        likelihood_ratio_bound = np.amax(likelihood_ratio)
        mylog.debug("LR: %s", likelihood_ratio_bound)
        if np.any(np.isnan(likelihood_ratio_bound) | np.isinf(likelihood_ratio_bound)):
            raise ValueError(
                "Cannot use the provided proposal because the likelihood ratio is"
                " degenerate in a subset of the abscissa.\nIs the proposal zero on the"
                " support of the likelihood?"
            )

        proposal *= likelihood_ratio_bound

    # @@ RESULT BUFFER MANAGEMENT @@ #
    # At this stage, we need to setup the result buffer / HDF5 dataset
    # to use.
    _expected_shape = (
        (n_samples, _number_of_dims) if _number_of_dims > 1 else (n_samples,)
    )
    if result_buffer is not None:
        # We got a result buffer from the input. We need to check that it is
        # the correct shape and then redirect to the correct C-level routine.
        if isinstance(result_buffer, (np.ndarray, h5py.Dataset)):
            # Validate that the buffer has a valid size.
            if np.any(
                [
                    result_buffer.shape[_a] < _expected_shape[_a]
                    for _a in range(_number_of_dims)
                ]
            ):
                raise ValueError(
                    f"The `result_buffer` input has shape ({result_buffer.shape}), but expected less than {(n_samples, _number_of_dims)}."
                )

            # Determine if its HDF5 or not.
            _is_hdf5 = isinstance(result_buffer, h5py.Dataset)
        else:
            raise ValueError(
                f"`result_buffer` is type {type(result_buffer)}, expected `np.ndarray` or `h5py.Dataset`."
            )
    else:
        # The result buffer is not provided, we can just generate it.
        result_buffer = np.empty(_expected_shape, dtype="f8", order="C")
        _is_hdf5 = False

    # @@ PASSING TO CYTHON / C LEVEL @@ #
    # At this stage in the execution, we are ready to just perform the exection
    # at the Cython and C level. Most of the code here determines which sup-procedure to
    # utilize.
    try:
        _sampler_key = (_number_of_dims, proposal is not None, _is_hdf5)
        _sampler = _rejection_sampler_map[_sampler_key]
    except KeyError:
        raise ValueError(
            f"No sampler available for {_number_of_dims} where prop={_sampler_key[1]} and hdf5={_sampler_key[2]}."
        )

    # correct the 1D case
    if _number_of_dims == 1:
        x = np.reshape(x, (x.size,), order="C")
        y = np.reshape(y, (y.size,), order="C")

    if proposal is not None:
        # noinspection PyTypeChecker
        _sampler(
            np.asarray(x, order="C", dtype="f8"),
            np.asarray(y, order="C", dtype="f8"),
            np.asarray(proposal_x, order="C", dtype="f8"),
            np.asarray(proposal, order="C", dtype="f8"),
            np.asarray(proposal_cdf, order="C", dtype="f8"),
            result_buffer,
            chunk_size=chunk_size,
            max_iterations=max_iterations,
            show_progress=show_progress,
            paxis=paxis,
        )
    else:
        # noinspection PyTypeChecker
        _sampler(
            np.asarray(x, order="C", dtype="f8"),
            np.asarray(y, order="C", dtype="f8"),
            result_buffer,
            chunk_size=chunk_size,
            max_iterations=max_iterations,
            show_progress=show_progress,
        )

    # Return the result.
    return result_buffer
