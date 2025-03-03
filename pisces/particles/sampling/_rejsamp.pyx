# cython: boundscheck=False, wraparound=False, cdivision=True
# ============================================================ #
# Optimized rejection samplers                                 #
#                                                              #
# The functions in this module provide C-level functions for   #
# sampling particles from specific density profiles. These     #
# should never be called directly as they have no handling     #
# for Jacobian scaling etc.                                    #
#                                                              #
# Written by: Eliza Diggins, copyright 2025                    #
# ============================================================ #

# ------------------------------------------- #
# Setup and Imports                           #
# ------------------------------------------- #
# _linterp provides access to C-level linear interpolation schemes
#   which can be used for sampling from discrete data.
#
# _invsamp provides access to C-level inversion sampling methods.

cimport numpy as np

from tqdm.auto import tqdm

from libc.stdlib cimport free, malloc

from pisces.particles.sampling cimport _invsamp
from pisces.utilities.math_utils cimport _linterp


# Import RNG methods from C standard library to ensure
# that we can avoid calls to the python layer.
# Also sets the random number seed.
cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil
srand48(-100)

# Define custom types.
ctypedef np.float64_t DTYPE_t
ctypedef np.uint64_t INT_TYPE_t
ctypedef np.uint8_t B_TYPE_t

# ------------------------------------------- #
# Generic Rejection Samplers                  #
# ------------------------------------------- #
# The generic rejection sampler core functions manage the basic protocol
# for rejection sampling:
#
# 1. Create the random sample of points.
# 2. Check each point.
# 3. Add points to the result buffer.
cdef unsigned long long _brejs1d_core(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    DTYPE_t[: ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
):
    """
    _brejs1d_core is the first of 3 generic 1-cycle rejection samplers. This is a core
    building block of later rejection sampling methodologies.

    Parameters
    ----------
    abscissa: (Nx,) array of x values.
    likelihood: (Nx,) array of likelihood values (max = 1)
    result_buffer: (N,) array into which values are placed.
    proposal_buffer: (N_chunk,3) array into which we place
        0: The x value of the proposal.
        1: The ordinate value of the proposal (to be compared against the likelihood).
        2: The likelihood at x.
    bounds: (2,) array of xmin and xmax.
    ridx: int the index into which new samples should be placed in the result buffer.

    Returns
    -------
    ridx: The new ridx value after running this onces.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i

    # Randomly sample the support of the distribution using random sampling.
    for i in range(chunksize):
        proposal_buffer[i, 0] = bounds[0] + (bounds[1] - bounds[0]) * drand48()

    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    _linterp.eval_clinterp1d(
        proposal_buffer[:, 0],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 2]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i,1] = drand48()

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i,1] <= proposal_buffer[i,2]:
            # We have an accepted case.
            result_buffer[ridx] = proposal_buffer[i,0]
            ridx += 1

    return ridx

cdef unsigned long long _brejs2d_core(
    DTYPE_t[:, :, ::1 ] abscissa,
    DTYPE_t[:, ::1 ] likelihood,
    DTYPE_t[:, ::1 ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
):
    """
    _brejs2d_core is the second of 3 generic 1-cycle rejection samplers. This is a core
    building block of later rejection sampling methodologies.

    Parameters
    ----------
    abscissa: (Nx,Ny,2) array of x and y values.
    likelihood: (Nx,Ny) array of likelihood values (max = 1)
    result_buffer: (N,2) array into which values are placed.
    proposal_buffer: (N_chunk,4) array into which we place
        0: The x value of the proposal.
        1: The y value of the proposal.
        2: The ordinate value of the proposal (to be compared against the likelihood).
        3: The likelihood at x.
    bounds: (4,) array of xmin, xmax, ymin, ymax.
    ridx: int the index into which new samples should be placed in the result buffer.

    Returns
    -------
    ridx: The new ridx value after running this onces.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i
    cdef unsigned int j

    # Randomly sample the support of the distribution using random sampling.
    for i in range(chunksize):
        for j in range(2):
            proposal_buffer[i, j] = bounds[2*j] + (bounds[(2*j+1)] - bounds[2*j]) * drand48()

    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    _linterp.eval_clinterp2d(
        proposal_buffer[:, :2],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 3]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i,2] = drand48()

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i,2] <= proposal_buffer[i,3]:
            # We have an accepted case.
            result_buffer[ridx,0] = proposal_buffer[i,  0]
            result_buffer[ridx, 1] = proposal_buffer[i, 1]
            ridx += 1

    return ridx

cdef unsigned long long _brejs3d_core(
    DTYPE_t[:, :, :, ::1 ] abscissa,
    DTYPE_t[:,:, ::1 ] likelihood,
    DTYPE_t[:, ::1 ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
):
    """
    _brejs3d_core is the third of 3 generic 1-cycle rejection samplers. This is a core
    building block of later rejection sampling methodologies.

    Parameters
    ----------
    abscissa: (Nx,Ny,Nz,3) array of x and y and z values.
    likelihood: (Nx,Ny,Nz) array of likelihood values (max = 1)
    result_buffer: (N,3) array into which values are placed.
    proposal_buffer: (N_chunk,5) array into which we place
        0: The x value of the proposal.
        1: The y value of the proposal.
        2: The z value of the proposal.
        3: The ordinate value of the proposal (to be compared against the likelihood).
        4: The likelihood at x.
    bounds: (6,) array of xmin, xmax, ymin, ymax, zmin, zmax.
    ridx: int the index into which new samples should be placed in the result buffer.

    Returns
    -------
    ridx: The new ridx value after running this onces.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i
    cdef unsigned int j

    # Randomly sample the support of the distribution using random sampling.
    for i in range(chunksize):
        for j in range(3):
            proposal_buffer[i, j] = bounds[2*j] + (bounds[(2*j+1)] - bounds[2*j]) * drand48()

    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    _linterp.eval_clinterp3d(
        proposal_buffer[:, :3],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 4]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i,3] = drand48()

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i,3] <= proposal_buffer[i,4]:
            # We have an accepted case.
            result_buffer[ridx,0] = proposal_buffer[i,  0]
            result_buffer[ridx, 1] = proposal_buffer[i, 1]
            result_buffer[ridx, 2] = proposal_buffer[i, 2]
            ridx += 1

    return ridx

# ------------------------------------------- #
# Proposal Rejection Samplers                 #
# ------------------------------------------- #
# Proposal rejection samplers allow the user to provide a 1D distribution as a
# sampling distribution in order to generate the sample of points in the abscissa.
#
# 1. Create the random sample of points.
# 2. Check each point.
# 3. Add points to the result buffer.
cdef unsigned long long _prejs1d_core(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    DTYPE_t[: ] proposal_abscissa,
    DTYPE_t[: ] proposal_likelihood,
    DTYPE_t[: ] proposal_cdf,
    DTYPE_t[: ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
    unsigned int paxis
):
    """
    _prejs3d_core is the more generic equivalent to _brejs1d_core, but now implements the
    generic proposal input.

    Parameters
    ----------
    abscissa: (Nx,) array of x values.
    likelihood: (Nx,) array of likelihood values (max = 1)
    proposal_abscissa: (Nx,) the abscissa of the proposal points.
    proposal_likelihood: (Nx,) the likelihood at each proposal point.
    proposal_cdf: (Nx,) The CDF for the proposal.
    result_buffer: (N,) array into which values are placed.
    proposal_buffer: (N_chunk,4) array into which we place
        0: The x value of the proposal.
        1: The sampled oordinate value. This is U[0,1]*proposal_likelihood
           at each point in the abscissa.
        2: The likelihood at x.
        3: The proposal likelihood at x.
    bounds: (2,) array of xmin, xmax.
    ridx: int the index into which new samples should be placed in the result buffer.
    paxis: int the index of the proposal (always zero here).
    Returns
    -------
    ridx: The new ridx value after running this onces.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i
    cdef unsigned int j
    # Randomly sample points from the proposed distribution in order to draw
    # the proposal sample from the domain.
    _invsamp._invcsamp_cdf(proposal_abscissa,
                           proposal_cdf,
                           proposal_buffer[:, 0])

    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    # Additionally, determine the proposal likelihood at each point.
    _linterp.eval_clinterp1d(
        proposal_buffer[:, 0],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 2]
    )
    _linterp.eval_clinterp1d(
        proposal_buffer[:, 0],
        proposal_abscissa,
        proposal_likelihood,
        bounds, # Only valid in 1D cause the bounds are the same.
        proposal_buffer[:, 3]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i,1] = drand48() * proposal_buffer[i,3]

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i,1] <= proposal_buffer[i,2]:
            # We have an accepted case.
            result_buffer[ridx] = proposal_buffer[i,0]
            ridx += 1

    return ridx

cdef unsigned long long _prejs2d_core(
    DTYPE_t[:, :, ::1] abscissa,
    DTYPE_t[:, ::1] likelihood,
    DTYPE_t[: ] proposal_abscissa,
    DTYPE_t[: ] proposal_likelihood,
    DTYPE_t[: ] proposal_cdf,
    DTYPE_t[:, ::1 ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
    unsigned int paxis,
):
    """
    _prejs3d_core is the more generic equivalent to _brejs1d_core, but now implements the
    generic proposal input.

    Parameters
    ----------
    abscissa: (Nx,Ny,2) array of x values.
    likelihood: (Nx,Ny) array of likelihood values (max = 1)
    proposal_abscissa: (Ni,) the abscissa of the proposal points.
    proposal_likelihood: (Ni,) the likelihood at each proposal point.
    proposal_cdf: (Ni,) The CDF for the proposal.
    result_buffer: (N,2) array into which values are placed.
    proposal_buffer: (N_chunk,5) array into which we place
        0: The x value of the proposal.
        1: The y value of the proposal.
        2: The sampled oordinate value. This is U[0,1]*proposal_likelihood
           at each point in the abscissa.
        3: The likelihood at x.
        4: The proposal likelihood at x.
    bounds: (4,) array of xmin, xmax, ymin, ymax.
    ridx: int the index into which new samples should be placed in the result buffer.

    Returns
    -------
    ridx: The new ridx value after running this onces.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i
    cdef unsigned int j

    # Randomly sample the support of the distribution using random sampling.
    # For the proposal axis, we need to take an inversion sample.
    for j in range(2):
        if paxis == j:
            _invsamp._invcsamp_cdf(proposal_abscissa,
                                   proposal_cdf,
                                   proposal_buffer[:, j])
        else:
            for i in range(chunksize):
                proposal_buffer[i, j] = bounds[2 * j] + (bounds[(2 * j + 1)] - bounds[2 * j]) * drand48()


    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    _linterp.eval_clinterp2d(
        proposal_buffer[:, :2],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 3]
    )
    _linterp.eval_clinterp1d(
        proposal_buffer[:, paxis],
        proposal_abscissa,
        proposal_likelihood,
        bounds[2*paxis:2*paxis+1], # Only valid in 1D cause the bounds are the same.
        proposal_buffer[:, 4]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i, 2] = drand48() * proposal_buffer[i,4]

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i, 2] <= proposal_buffer[i, 3]:
            # We have an accepted case.
            result_buffer[ridx, 0] = proposal_buffer[i, 0]
            result_buffer[ridx, 1] = proposal_buffer[i, 1]
            ridx += 1

    return ridx

cdef unsigned long long _prejs3d_core(
    DTYPE_t[:, :, :, ::1] abscissa,
    DTYPE_t[:,:, ::1] likelihood,
    DTYPE_t[: ] proposal_abscissa,
    DTYPE_t[: ] proposal_likelihood,
    DTYPE_t[: ] proposal_cdf,
    DTYPE_t[:, ::1 ] result_buffer,
    DTYPE_t[:, ::1 ] proposal_buffer,
    DTYPE_t[: ] bounds,
    unsigned long long ridx,
    unsigned int paxis,
):
    """
    _prejs3d_core is the more generic equivalent to _brejs1d_core, but now implements the
    generic proposal input.

    Parameters
    ----------
    abscissa: (Nx,Ny,Nz,3) array of x values.
    likelihood: (Nx,Ny,Nz) array of likelihood values (max = 1)
    proposal_abscissa: (Ni,) the abscissa of the proposal points.
    proposal_likelihood: (Ni,) the likelihood at each proposal point.
    proposal_cdf: (Ni,) The CDF for the proposal.
    result_buffer: (N,3) array into which values are placed.
    proposal_buffer: (N_chunk,6) array into which we place
        0: The x value of the proposal.
        1: The y value of the proposal.
        2: The z value of the proposal.
        3: The sampled oordinate value. This is U[0,1]*proposal_likelihood
           at each point in the abscissa.
        4: The likelihood at x.
        5: The proposal likelihood at x.
    bounds: (6,) array of xmin, xmax, ymin, ymax, zmin, zmax
    ridx: int the index into which new samples should be placed in the result buffer.

    Returns
    -------
    ridx: The new ridx value after running this once.
    """
    # Declare the chunk size and the iteration index variable
    cdef Py_ssize_t chunksize = proposal_buffer.shape[0], numsamples = result_buffer.shape[0]
    cdef unsigned long i
    cdef unsigned int j

    # Randomly sample the support of the distribution using random sampling.
    # For the proposal axis, we need to take an inversion sample.
    for j in range(3):
        if paxis == j:
            _invsamp._invcsamp_cdf(proposal_abscissa,
                                   proposal_cdf,
                                   proposal_buffer[:, j])
        else:
            for i in range(chunksize):
                proposal_buffer[i, j] = bounds[2 * j] + (bounds[(2 * j + 1)] - bounds[2 * j]) * drand48()


    # Interpolated over the support points to determine the value of the
    # likelihood at each point.
    _linterp.eval_clinterp3d(
        proposal_buffer[:, :3],
        abscissa,
        likelihood,
        bounds,
        proposal_buffer[:, 4]
    )
    _linterp.eval_clinterp1d(
        proposal_buffer[:, paxis],
        proposal_abscissa,
        proposal_likelihood,
        bounds[2*paxis:2*paxis+1], # Only valid in 1D cause the bounds are the same.
        proposal_buffer[:, 5]
    )

    # Cycle through each of the samples and determine if we are going to accept or
    # reject this proposed sample point. If so, we need to add it to the result_buffer
    # and increment ridx.
    for i in range(chunksize):
        # Check if we are ready to break the cycle.
        if ridx >= numsamples:
            return ridx

        # Determine the randomly selected ordinate to use for
        # the rejection / acceptance process.
        # - In the generic case, we simply take a random var between 0 and 1 since
        #   the likelihood is normalized.
        proposal_buffer[i, 3] = drand48() * proposal_buffer[i,5]

        # Check if the proposal here is a valid proposal. If so,
        # we just need to add the proposal to the sample.
        if proposal_buffer[i, 3] <= proposal_buffer[i, 4]:
            # We have an accepted case.
            result_buffer[ridx, 0] = proposal_buffer[i, 0]
            result_buffer[ridx, 1] = proposal_buffer[i, 1]
            result_buffer[ridx, 2] = proposal_buffer[i, 2]
            ridx += 1

    return ridx

# -------------------------------------------- #
# Chunking Wrappers                            #
# -------------------------------------------- #
cdef void _brejs1d(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    DTYPE_t[: ] result_buffer,
    unsigned long chunk_size = 10_000,
    unsigned long max_iterations = 10_000,
    bint show_progress = False):
    """
    Perform rejection sampling using 1D interpolation.

    This function implements a rejection sampling algorithm to generate samples
    from a given likelihood function, which is defined at discrete points
    (`abscissa`). The function uses linear interpolation to evaluate the
    likelihood at randomly sampled points and accepts samples based on a
    rejection condition.

    Parameters
    ----------
    abscissa : DTYPE_t[:]
        A 1D array of x-values (support points) at which the likelihood function
        is defined. This array should be strictly increasing.

    likelihood : DTYPE_t[:]
        A 1D array of corresponding likelihood values for each `abscissa` value.

    result_buffer : DTYPE_t[:]
        A 1D array where the generated samples are stored. The number of samples
        to generate is determined by the size of `result_buffer`.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops in cases where rejection is high.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function generates uniform random samples within the range of `abscissa`
      and evaluates their likelihood using linear interpolation.
    - A rejection sampling criterion is applied: a sample is accepted if a uniform
      random number is less than the interpolated likelihood.
    - The algorithm iterates until the required number of samples (`result_buffer.size`)
      is generated or until `max_iterations` is reached.
    - The function allocates temporary memory for internal computations and
      frees it before returning.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0], nsupp = abscissa.shape[0]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[2] bounds = [abscissa[0],abscissa[nsupp-1]]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(3* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :3]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (1D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _brejs1d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev)

        # Compute the rate and update the
        # progress bar.
        rate = <double>(samp_indx-samp_indx_prev) / chunk_size

        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (1D) - AR ~ {rate * 100:.2f}%")

        samp_indx_prev = samp_indx

        # Check for a breaking condition.
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

cdef void _brejs2d(
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] likelihood,
        DTYPE_t[:, ::1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Perform rejection sampling using 1D interpolation.

    This function implements a rejection sampling algorithm to generate samples
    from a given likelihood function, which is defined at discrete points
    (`abscissa`). The function uses linear interpolation to evaluate the
    likelihood at randomly sampled points and accepts samples based on a
    rejection condition.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, ::1]
        The abscissa from which to interpolate the likelihood function. This should be a
        DTYPE_t array of shape (Nx,Ny,2) containing the x and y positions.

    likelihood : DTYPE_t[:, :]
        A 2D array of corresponding likelihood values for each `abscissa` value.

    result_buffer : DTYPE_t[:, ::1]
        A 2D array where the generated samples are stored. The number of samples
        to generate is determined by the size of `result_buffer`.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops in cases where rejection is high.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function generates uniform random samples within the range of `abscissa`
      and evaluates their likelihood using linear interpolation.
    - A rejection sampling criterion is applied: a sample is accepted if a uniform
      random number is less than the interpolated likelihood.
    - The algorithm iterates until the required number of samples (`result_buffer.size`)
      is generated or until `max_iterations` is reached.
    - The function allocates temporary memory for internal computations and
      frees it before returning.

    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[4] bounds = [
        abscissa[0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0],    # x_max
        abscissa[0, 0, 1],       # y_min
        abscissa[0, Ny-1, 1],    # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(4* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :4]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (2D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _brejs2d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev)
        # Compute the rate and update the
        # progress bar.
        rate = <double> (samp_indx - samp_indx_prev) / chunk_size


        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (2D) - AR ~ {rate * 100:.2f}%")

        samp_indx_prev = samp_indx
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

cdef void _brejs3d(
        DTYPE_t[: , : , : , ::1] abscissa,
        DTYPE_t[: , : , ::1] likelihood,
        DTYPE_t[: , ::1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Perform rejection sampling using 3D interpolation (trilinear).

    This function implements a rejection sampling algorithm to generate samples
    from a given 3D likelihood function, which is defined on a rectilinear grid
    of coordinates `abscissa`. The function uses trilinear interpolation to evaluate
    the likelihood at randomly sampled points (x, y, z) within the bounding region,
    then accepts or rejects these proposals based on a uniform random criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, :, ::1]
        A 4D array of shape (Nx, Ny, Nz, 3), where
        `abscissa[i, j, k, 0]` = x-coordinate,
        `abscissa[i, j, k, 1]` = y-coordinate,
        `abscissa[i, j, k, 2]` = z-coordinate.
        This array should be strictly increasing along each dimension
        (x, then y, then z).

    likelihood : DTYPE_t[:, :, ::1]
        A 3D array of shape (Nx, Ny, Nz) giving the likelihood value at each
        grid point `(i, j, k)`.

    result_buffer : DTYPE_t[:, ::1]
        A 2D array of shape (N_samples, 3), where the accepted (x, y, z) samples
        are stored. The number of samples to generate is determined by
        `result_buffer.shape[0]`.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops if the acceptance rate is extremely low.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function randomly samples (x, y, z) within the bounding box inferred from
      the minimum and maximum corners of `abscissa`.
    - Trilinear interpolation is used (via `eval_clinterp3d`) to evaluate the
      likelihood at each proposed point.
    - A uniform random draw `U ~ [0, 1]` is used to accept the sample if
      `U < likelihood_interpolated`.
    - Repeats until `result_buffer.shape[0]` samples are accepted, or
      `max_iterations` is reached.
    - Allocates temporary memory for proposals and frees it before returning.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef Py_ssize_t Nz = abscissa.shape[2]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[6] bounds = [
        abscissa[0, 0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0, 0],    # x_max
        abscissa[0, 0, 0, 1],       # y_min
        abscissa[0, Ny-1, 0, 1],    # y_max
        abscissa[0, 0, 0, 2],  # y_min
        abscissa[0, 0, Nz-1, 2],  # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(5* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :5]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (3D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _brejs3d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev)

        # Compute the rate and update the
        # progress bar.
        rate = <double> (samp_indx - samp_indx_prev) / chunk_size


        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (3D) - AR ~ {rate * 100:.2f}%")
        samp_indx_prev = samp_indx
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

cdef void _prejs1d(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    DTYPE_t[: ] proposal_abscissa,
    DTYPE_t[: ] proposal_likelihood,
    DTYPE_t[: ] proposal_cdf,
    DTYPE_t[: ] result_buffer,
    unsigned long chunk_size = 10_000,
    unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Perform 1D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target likelihood function defined on a 1D grid.
    A proposal distribution is used to improve the sampling efficiency. Inverse transform sampling
    is applied to the proposal distribution (defined by its support, likelihood, and CDF) to obtain candidate
    abscissa values. The function then interpolates both the target and proposal likelihoods and accepts
    candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:]
        1D array of x-values where the target likelihood is defined. Must be strictly increasing.
    likelihood : DTYPE_t[:]
        1D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : DTYPE_t[:]
        Pre-allocated 1D array where accepted sample positions will be stored.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : whether to display a tqdm progress bar in Python.

    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    This function uses C-level memory management and linear interpolation for efficient execution.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0], nsupp = abscissa.shape[0]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[2] bounds = [abscissa[0],abscissa[nsupp-1]]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(4* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :4]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (1D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _prejs1d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev,
                    0)

        # Compute the rate and update the
        # progress bar.
        rate = <double> (samp_indx - samp_indx_prev) / chunk_size


        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (1D) - AR ~ {rate * 100:.2f}%")
        samp_indx_prev = samp_indx
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

cdef void _prejs2d(
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] likelihood,
        DTYPE_t[:] proposal_abscissa,
        DTYPE_t[:] proposal_likelihood,
        DTYPE_t[:] proposal_cdf,
        DTYPE_t[:, ::1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Perform 2D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target 2D likelihood function defined on a grid using a proposal
    distribution that is non-uniform along one specified axis. Inverse transform sampling is applied along
    the proposal axis (`paxis`), while the other axis is sampled uniformly. The function interpolates both the
    target and proposal likelihoods and accepts candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, ::1]
        3D array of shape (Nx, Ny, 2) containing grid coordinates for the target likelihood.
        Each element holds the (x, y) coordinates.
    likelihood : DTYPE_t[:, ::1]
        2D array of likelihood values of shape (Nx, Ny) corresponding to the grid points.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : DTYPE_t[:, ::1]
        Pre-allocated 2D array of shape (N_samples, 2) where accepted (x, y) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index (0 or 1) to use for inverse sampling in the proposal distribution (default: 1).
    show_progress : whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    Linear interpolation is used for evaluating both target and proposal likelihoods.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[4] bounds = [
        abscissa[0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0],    # x_max
        abscissa[0, 0, 1],       # y_min
        abscissa[0, Ny-1, 1],    # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(5* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :5]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (2D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _prejs2d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev,
                    paxis)

        # Compute the rate and update the
        # progress bar.
        rate = <double> (samp_indx - samp_indx_prev) / chunk_size


        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (2D) - AR ~ {rate * 100:.2f}%")
        samp_indx_prev = samp_indx
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

cdef void _prejs3d(
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:,:,  ::1] likelihood,
        DTYPE_t[:] proposal_abscissa,
        DTYPE_t[:] proposal_likelihood,
        DTYPE_t[:] proposal_cdf,
        DTYPE_t[:, ::1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Perform 3D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target 3D likelihood function defined on a rectilinear grid.
    A proposal distribution that is non-uniform along one axis is used to improve sampling efficiency.
    Inverse transform sampling is applied on the proposal axis (`paxis`), while the other axes are sampled uniformly.
    The function uses trilinear interpolation for the target likelihood and linear interpolation for the proposal likelihood,
    accepting candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, :, ::1]
        4D array of shape (Nx, Ny, Nz, 3) containing grid coordinates for the target likelihood.
        Each element provides the (x, y, z) coordinates.
    likelihood : DTYPE_t[:,:,  ::1]
        3D array of likelihood values of shape (Nx, Ny, Nz) corresponding to the grid points.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : DTYPE_t[:, ::1]
        Pre-allocated 2D array of shape (N_samples, 3) where accepted (x, y, z) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index (0, 1, or 2) used for inverse transform sampling (default: 1).
    show_progress : whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    Trilinear interpolation is used for the target likelihood evaluation, while linear interpolation is used
    for the proposal likelihood.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = result_buffer.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef Py_ssize_t Nz = abscissa.shape[2]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,samp_indx_prev = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[6] bounds = [
        abscissa[0, 0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0, 0],    # x_max
        abscissa[0, 0, 0, 1],       # y_min
        abscissa[0, Ny-1, 0, 1],    # y_max
        abscissa[0, 0, 0, 2],  # y_min
        abscissa[0, 0, Nz-1, 2],  # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(6* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :6]>proposal

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (3D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        samp_indx = _prejs3d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      samp_indx_prev,
                    paxis)

        # Compute the rate and update the
        # progress bar.
        rate = <double> (samp_indx - samp_indx_prev) / chunk_size


        if show_progress:
            pbar.update(samp_indx-samp_indx_prev)
            pbar.set_description(f"Sampling (3D) - AR ~ {rate * 100:.2f}%")
        samp_indx_prev = samp_indx
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)

    return

# -------------------------------------------- #
# HDF5 Wrappers                                #
# -------------------------------------------- #
cdef void _brejs1d_hdf5(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    object dataset,
    unsigned long chunk_size = 10_000,
    unsigned long max_iterations = 10_000,
    bint show_progress = False):
    """
    Perform rejection sampling using 1D interpolation.

    This function implements a rejection sampling algorithm to generate samples
    from a given likelihood function, which is defined at discrete points
    (`abscissa`). The function uses linear interpolation to evaluate the
    likelihood at randomly sampled points and accepts samples based on a
    rejection condition.

    Parameters
    ----------
    abscissa : DTYPE_t[:]
        A 1D array of x-values (support points) at which the likelihood function
        is defined. This array should be strictly increasing.

    likelihood : DTYPE_t[:]
        A 1D array of corresponding likelihood values for each `abscissa` value.

    dataset: HDF5 Dataset
        The dataset into which to deposit the data.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops in cases where rejection is high.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function generates uniform random samples within the range of `abscissa`
      and evaluates their likelihood using linear interpolation.
    - A rejection sampling criterion is applied: a sample is accepted if a uniform
      random number is less than the interpolated likelihood.
    - The algorithm iterates until the required number of samples (`result_buffer.size`)
      is generated or until `max_iterations` is reached.
    - The function allocates temporary memory for internal computations and
      frees it before returning.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0], nsupp = abscissa.shape[0]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0, new_this_chunk=0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[2] bounds = [abscissa[0],abscissa[nsupp-1]]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(3* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :3]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(chunk_size*sizeof(double))
    cdef double[:] result_buffer = <double[:chunk_size]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (1D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic and increment the
        # sample index.
        new_this_chunk = _brejs1d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0)

        # Write the new data to the HDF5 buffer.
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        if dataset.ndim > 1:
            dataset[samp_indx:samp_indx + new_this_chunk,:1] = result_buffer[:new_this_chunk]
        else:
            dataset[samp_indx:samp_indx + new_this_chunk] = result_buffer[:new_this_chunk]

        # Compute the rate and update the
        # progress bar.
        rate = <double>(new_this_chunk) / chunk_size

        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (1D) - AR ~ {rate * 100:.2f}%")

        samp_indx += new_this_chunk

        # Check for a breaking condition.
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return

cdef void _brejs2d_hdf5(
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] likelihood,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Perform rejection sampling using 1D interpolation.

    This function implements a rejection sampling algorithm to generate samples
    from a given likelihood function, which is defined at discrete points
    (`abscissa`). The function uses linear interpolation to evaluate the
    likelihood at randomly sampled points and accepts samples based on a
    rejection condition.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, ::1]
        The abscissa from which to interpolate the likelihood function. This should be a
        DTYPE_t array of shape (Nx,Ny,2) containing the x and y positions.

    likelihood : DTYPE_t[:, :]
        A 2D array of corresponding likelihood values for each `abscissa` value.

    dataset: HDF5 Dataset
        The dataset into which to deposit the data.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops in cases where rejection is high.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function generates uniform random samples within the range of `abscissa`
      and evaluates their likelihood using linear interpolation.
    - A rejection sampling criterion is applied: a sample is accepted if a uniform
      random number is less than the interpolated likelihood.
    - The algorithm iterates until the required number of samples (`result_buffer.size`)
      is generated or until `max_iterations` is reached.
    - The function allocates temporary memory for internal computations and
      frees it before returning.

    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0,new_this_chunk = 0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[4] bounds = [
        abscissa[0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0],    # x_max
        abscissa[0, 0, 1],       # y_min
        abscissa[0, Ny-1, 1],    # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(4* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :4]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(2*chunk_size*sizeof(double))
    cdef double[:,::1] result_buffer = <double[:chunk_size,:2]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (2D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        new_this_chunk = _brejs2d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0)

        # Write data to the dataset buffer
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        dataset[samp_indx:samp_indx+new_this_chunk,:2] = result_buffer[:new_this_chunk,:]


        # Compute the rate and update the
        # progress bar.
        rate = <double> (new_this_chunk) / chunk_size


        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (2D) - AR ~ {rate * 100:.2f}%")
        samp_indx += new_this_chunk
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return

cdef void _brejs3d_hdf5(
        DTYPE_t[: , : , : , ::1] abscissa,
        DTYPE_t[: , : , ::1] likelihood,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Perform rejection sampling using 3D interpolation (trilinear).

    This function implements a rejection sampling algorithm to generate samples
    from a given 3D likelihood function, which is defined on a rectilinear grid
    of coordinates `abscissa`. The function uses trilinear interpolation to evaluate
    the likelihood at randomly sampled points (x, y, z) within the bounding region,
    then accepts or rejects these proposals based on a uniform random criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, :, ::1]
        A 4D array of shape (Nx, Ny, Nz, 3), where
        `abscissa[i, j, k, 0]` = x-coordinate,
        `abscissa[i, j, k, 1]` = y-coordinate,
        `abscissa[i, j, k, 2]` = z-coordinate.
        This array should be strictly increasing along each dimension
        (x, then y, then z).

    likelihood : DTYPE_t[:, :, ::1]
        A 3D array of shape (Nx, Ny, Nz) giving the likelihood value at each
        grid point `(i, j, k)`.

    dataset: HDF5 Dataset
        The dataset into which to deposit the data.

    chunk_size : unsigned long, optional
        The number of proposals to generate per iteration (default: 10,000).
        Larger values may improve efficiency but use more memory.

    max_iterations : unsigned long, optional
        The maximum number of iterations to attempt before stopping (default: 10,000).
        This prevents infinite loops if the acceptance rate is extremely low.

    show_progress : whether to display a tqdm progress bar in Python.

    Notes
    -----
    - The function randomly samples (x, y, z) within the bounding box inferred from
      the minimum and maximum corners of `abscissa`.
    - Trilinear interpolation is used (via `eval_clinterp3d`) to evaluate the
      likelihood at each proposed point.
    - A uniform random draw `U ~ [0, 1]` is used to accept the sample if
      `U < likelihood_interpolated`.
    - Repeats until `result_buffer.shape[0]` samples are accepted, or
      `max_iterations` is reached.
    - Allocates temporary memory for proposals and frees it before returning.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef Py_ssize_t Nz = abscissa.shape[2]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0, new_this_chunk=0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[6] bounds = [
        abscissa[0, 0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0, 0],    # x_max
        abscissa[0, 0, 0, 1],       # y_min
        abscissa[0, Ny-1, 0, 1],    # y_max
        abscissa[0, 0, 0, 2],  # y_min
        abscissa[0, 0, Nz-1, 2],  # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(5* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :5]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(3*chunk_size*sizeof(double))
    cdef double[:,::1] result_buffer = <double[:chunk_size,:3]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (3D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        new_this_chunk = _brejs3d_core(abscissa,
                      likelihood,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0)

        # Write data to the dataset buffer
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        dataset[samp_indx:samp_indx+new_this_chunk,:3] = result_buffer[:new_this_chunk,:]

        # Compute the rate and update the
        # progress bar.
        rate = <double> (new_this_chunk) / chunk_size

        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (3D) - AR ~ {rate * 100:.2f}%")
        samp_indx += new_this_chunk

        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return

cdef void _prejs1d_hdf5(
    DTYPE_t[: ] abscissa,
    DTYPE_t[: ] likelihood,
    DTYPE_t[: ] proposal_abscissa,
    DTYPE_t[: ] proposal_likelihood,
    DTYPE_t[: ] proposal_cdf,
    object dataset,
    unsigned long chunk_size = 10_000,
    unsigned long max_iterations = 10_000,
    bint show_progress = False):
    """
    Perform 1D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target likelihood function defined on a 1D grid.
    A proposal distribution is used to improve the sampling efficiency. Inverse transform sampling
    is applied to the proposal distribution (defined by its support, likelihood, and CDF) to obtain candidate
    abscissa values. The function then interpolates both the target and proposal likelihoods and accepts
    candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:]
        1D array of x-values where the target likelihood is defined. Must be strictly increasing.
    likelihood : DTYPE_t[:]
        1D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : whether to display a tqdm progress bar in Python.

    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    This function uses C-level memory management and linear interpolation for efficient execution.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0], nsupp = abscissa.shape[0]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0, new_this_chunk=0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[2] bounds = [abscissa[0],abscissa[nsupp-1]]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(4* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :4]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(chunk_size*sizeof(double))
    cdef double[:] result_buffer = <double[:chunk_size]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (1D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        new_this_chunk = _prejs1d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0,
                    0)

        # Write data to the dataset buffer
        # Write the new data to the HDF5 buffer.
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        if dataset.ndim > 1:
            dataset[samp_indx:samp_indx + new_this_chunk,:1] = result_buffer[:new_this_chunk]
        else:
            dataset[samp_indx:samp_indx + new_this_chunk] = result_buffer[:new_this_chunk]

        # Compute the rate and update the
        # progress bar.
        rate = <double> (new_this_chunk) / chunk_size

        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (1D) - AR ~ {rate * 100:.2f}%")
        samp_indx += new_this_chunk
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return

cdef void _prejs2d_hdf5(
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] likelihood,
        DTYPE_t[:] proposal_abscissa,
        DTYPE_t[:] proposal_likelihood,
        DTYPE_t[:] proposal_cdf,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Perform 2D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target 2D likelihood function defined on a grid using a proposal
    distribution that is non-uniform along one specified axis. Inverse transform sampling is applied along
    the proposal axis (`paxis`), while the other axis is sampled uniformly. The function interpolates both the
    target and proposal likelihoods and accepts candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, ::1]
        3D array of shape (Nx, Ny, 2) containing grid coordinates for the target likelihood.
        Each element holds the (x, y) coordinates.
    likelihood : DTYPE_t[:, ::1]
        2D array of likelihood values of shape (Nx, Ny) corresponding to the grid points.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index (0 or 1) to use for inverse sampling in the proposal distribution (default: 1).
    show_progress : whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    Linear interpolation is used for evaluating both target and proposal likelihoods.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0, new_this_chunk=0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[4] bounds = [
        abscissa[0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0],    # x_max
        abscissa[0, 0, 1],       # y_min
        abscissa[0, Ny-1, 1],    # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(5* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :5]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(2*chunk_size*sizeof(double))
    cdef double[:,::1] result_buffer = <double[:chunk_size,:2]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (2D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        new_this_chunk = _prejs2d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0,
                    paxis)

        # Write data to the dataset buffer
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        dataset[samp_indx:samp_indx+new_this_chunk,:2] = result_buffer[:new_this_chunk,:]

        # Compute the rate and update the
        # progress bar.
        rate = <double> (new_this_chunk) / chunk_size


        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (2D) - AR ~ {rate * 100:.2f}%")
        samp_indx += new_this_chunk
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return

cdef void _prejs3d_hdf5(
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:,:,  ::1] likelihood,
        DTYPE_t[:] proposal_abscissa,
        DTYPE_t[:] proposal_likelihood,
        DTYPE_t[:] proposal_cdf,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Perform 3D proposal rejection sampling using a non-uniform proposal distribution.

    This function generates samples from a target 3D likelihood function defined on a rectilinear grid.
    A proposal distribution that is non-uniform along one axis is used to improve sampling efficiency.
    Inverse transform sampling is applied on the proposal axis (`paxis`), while the other axes are sampled uniformly.
    The function uses trilinear interpolation for the target likelihood and linear interpolation for the proposal likelihood,
    accepting candidates based on a rejection criterion.

    Parameters
    ----------
    abscissa : DTYPE_t[:, :, :, ::1]
        4D array of shape (Nx, Ny, Nz, 3) containing grid coordinates for the target likelihood.
        Each element provides the (x, y, z) coordinates.
    likelihood : DTYPE_t[:,:,  ::1]
        3D array of likelihood values of shape (Nx, Ny, Nz) corresponding to the grid points.
    proposal_abscissa : DTYPE_t[:]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : DTYPE_t[:]
        1D array of likelihood values for the proposal distribution corresponding to `proposal_abscissa`.
    proposal_cdf : DTYPE_t[:]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index (0, 1, or 2) used for inverse transform sampling (default: 1).
    show_progress : whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Accepted samples are written into `result_buffer` in place.

    Notes
    -----
    Trilinear interpolation is used for the target likelihood evaluation, while linear interpolation is used
    for the proposal likelihood.
    """
    # Declare basic variables.
    cdef Py_ssize_t nsamp = dataset.shape[0]
    cdef Py_ssize_t Nx = abscissa.shape[0]
    cdef Py_ssize_t Ny = abscissa.shape[1]
    cdef Py_ssize_t Nz = abscissa.shape[2]
    cdef unsigned long iter_indx = 0
    cdef unsigned long long samp_indx = 0, new_this_chunk=0
    cdef double rate = 0

    # Manage boundaries. We use the abscissa to identify the
    # bounds.
    cdef double[6] bounds = [
        abscissa[0, 0, 0, 0],       # x_min
        abscissa[Nx-1, 0, 0, 0],    # x_max
        abscissa[0, 0, 0, 1],       # y_min
        abscissa[0, Ny-1, 0, 1],    # y_max
        abscissa[0, 0, 0, 2],  # y_min
        abscissa[0, 0, Nz-1, 2],  # y_max
    ]

    # Allocate the proposal space. This is a chunk_size x 3 array which contains
    # 1. The randomly sampled position on the x axis.
    # 2. The randomly sampled position on the y axis.
    # 3. The interpolated y value to compare against 2.
    cdef double * proposal = <double *> malloc(6* chunk_size * sizeof(double))
    cdef double[:, ::1] proposal_buffer = <double[:chunk_size, :6]>proposal

    # Define the intermediate result buffer to hold the values
    # This gets the values from the C-level and then dumps them
    # to the HDF5 file.
    cdef double * result = <double *> malloc(3*chunk_size*sizeof(double))
    cdef double[:,::1] result_buffer = <double[:chunk_size,:3]>result

    # Setup the progress bar if necessary.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(total=nsamp, desc="Sampling (3D)", unit="samples",leave=False)

    # Iterate through values up to the maximum number of iterations
    while iter_indx < max_iterations:
        # Increment the iteration flag.
        iter_indx += 1

        # Call to the core logic.
        new_this_chunk = _prejs3d_core(abscissa,
                      likelihood,
                      proposal_abscissa,
                      proposal_likelihood,
                      proposal_cdf,
                      result_buffer,
                      proposal_buffer,
                      bounds,
                      0,
                    paxis)

        # Write data to the dataset buffer
        new_this_chunk = min(nsamp-samp_indx,new_this_chunk)
        dataset[samp_indx:samp_indx+new_this_chunk,:3] = result_buffer[:new_this_chunk,:]

        # Compute the rate and update the
        # progress bar.
        rate = <double> (new_this_chunk) / chunk_size

        if show_progress:
            pbar.update(new_this_chunk)
            pbar.set_description(f"Sampling (3D) - AR ~ {rate * 100:.2f}%")
        samp_indx += new_this_chunk
        if samp_indx >= nsamp:
            break

    if show_progress:
        pbar.close()

    # Release memory space
    free(proposal)
    free(result)

    return
# -------------------------------------------- #
# Python Wrappers                              #
# -------------------------------------------- #
cpdef prejs1d(
        np.ndarray[DTYPE_t, ndim=1] abscissa,
        np.ndarray[DTYPE_t, ndim=1] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        np.ndarray[DTYPE_t, ndim=1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        int paxis = 0,
        bint show_progress = False):
    """
    Python wrapper for 1D proposal rejection sampling.

    This function exposes the C-level `_prejs1d` function to Python. It generates samples from a target
    likelihood using a non-uniform proposal distribution defined by its support, likelihood, and CDF.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array of x-values for the target likelihood function.
    likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : np.ndarray[DTYPE_t, ndim=1]
        Pre-allocated array where accepted sample positions will be stored.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    paxis : unsigned int, optional
        Axis index used for inverse transform sampling (default: 1).

    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs1d(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef prejs2d(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Python wrapper for 2D proposal rejection sampling.

    This function exposes the C-level `_prejs2d` to Python. It generates samples from a target 2D likelihood
    function using a proposal distribution with non-uniform characteristics along the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        3D array of shape (Nx, Ny, 2) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=2]
        2D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        Pre-allocated 2D array of shape (N_samples, 2) where accepted (x, y) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index used for inverse transform sampling (default: 1).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs2d(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    paxis=paxis,
                    show_progress=show_progress)


cpdef prejs3d(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Python wrapper for 3D proposal rejection sampling.

    This function exposes the C-level `_prejs3d` function to Python. It generates samples from a target
    3D likelihood function using a non-uniform proposal distribution with inverse sampling on the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        4D array of shape (Nx, Ny, Nz, 3) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=3]
        3D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        Pre-allocated 2D array of shape (N_samples, 3) where accepted (x, y, z) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index used for inverse transform sampling (default: 1).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs3d(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    paxis=paxis,
                    show_progress=show_progress)


cpdef brejs1d(
        np.ndarray[DTYPE_t, ndim=1] abscissa,
        np.ndarray[DTYPE_t, ndim=1] likelihood,
        np.ndarray[DTYPE_t, ndim=1] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 1D proposal rejection sampling.

    This function exposes the C-level `_prejs1d` function to Python. It generates samples from a target
    likelihood using a non-uniform proposal distribution defined by its support, likelihood, and CDF.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array of x-values for the target likelihood function.
    likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values corresponding to `abscissa`.
    result_buffer : np.ndarray[DTYPE_t, ndim=1]
        Pre-allocated array where accepted sample positions will be stored.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs1d(abscissa,
                    likelihood,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef brejs2d(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] likelihood,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 2D proposal rejection sampling.

    This function exposes the C-level `_prejs2d` to Python. It generates samples from a target 2D likelihood
    function using a proposal distribution with non-uniform characteristics along the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        3D array of shape (Nx, Ny, 2) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=2]
        2D array of likelihood values corresponding to `abscissa`.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        Pre-allocated 2D array of shape (N_samples, 2) where accepted (x, y) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs2d(abscissa,
                    likelihood,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef brejs3d(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] likelihood,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 3D proposal rejection sampling.

    This function exposes the C-level `_prejs3d` function to Python. It generates samples from a target
    3D likelihood function using a non-uniform proposal distribution with inverse sampling on the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        4D array of shape (Nx, Ny, Nz, 3) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=3]
        3D array of likelihood values corresponding to `abscissa`.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        Pre-allocated 2D array of shape (N_samples, 3) where accepted (x, y, z) samples will be stored.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs3d(abscissa,
                    likelihood,
                    result_buffer,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef prejs1d_hdf5(
        np.ndarray[DTYPE_t, ndim=1] abscissa,
        np.ndarray[DTYPE_t, ndim=1] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 1D proposal rejection sampling.

    This function exposes the C-level `_prejs1d` function to Python. It generates samples from a target
    likelihood using a non-uniform proposal distribution defined by its support, likelihood, and CDF.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array of x-values for the target likelihood function.
    likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs1d_hdf5(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef prejs2d_hdf5(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Python wrapper for 2D proposal rejection sampling.

    This function exposes the C-level `_prejs2d` to Python. It generates samples from a target 2D likelihood
    function using a proposal distribution with non-uniform characteristics along the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        3D array of shape (Nx, Ny, 2) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=2]
        2D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index used for inverse transform sampling (default: 1).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs2d_hdf5(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    paxis=paxis,
                    show_progress=show_progress)


cpdef prejs3d_hdf5(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_abscissa,
        np.ndarray[DTYPE_t, ndim=1] proposal_likelihood,
        np.ndarray[DTYPE_t, ndim=1] proposal_cdf,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        unsigned int paxis = 1,
        bint show_progress = False):
    """
    Python wrapper for 3D proposal rejection sampling.

    This function exposes the C-level `_prejs3d` function to Python. It generates samples from a target
    3D likelihood function using a non-uniform proposal distribution with inverse sampling on the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        4D array of shape (Nx, Ny, Nz, 3) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=3]
        3D array of likelihood values corresponding to `abscissa`.
    proposal_abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array defining the support of the proposal distribution along the proposal axis.
    proposal_likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values for the proposal distribution.
    proposal_cdf : np.ndarray[DTYPE_t, ndim=1]
        1D array representing the cumulative distribution function of the proposal distribution.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    paxis : unsigned int, optional
        Axis index used for inverse transform sampling (default: 1).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _prejs3d_hdf5(abscissa,
                    likelihood,
                    proposal_abscissa,
                    proposal_likelihood,
                    proposal_cdf,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    paxis=paxis,
                    show_progress=show_progress)


cpdef brejs1d_hdf5(
        np.ndarray[DTYPE_t, ndim=1] abscissa,
        np.ndarray[DTYPE_t, ndim=1] likelihood,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 1D proposal rejection sampling.

    This function exposes the C-level `_prejs1d` function to Python. It generates samples from a target
    likelihood using a non-uniform proposal distribution defined by its support, likelihood, and CDF.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=1]
        1D array of x-values for the target likelihood function.
    likelihood : np.ndarray[DTYPE_t, ndim=1]
        1D array of likelihood values corresponding to `abscissa`.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals to generate per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs1d_hdf5(abscissa,
                    likelihood,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef brejs2d_hdf5(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] likelihood,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 2D proposal rejection sampling.

    This function exposes the C-level `_prejs2d` to Python. It generates samples from a target 2D likelihood
    function using a proposal distribution with non-uniform characteristics along the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        3D array of shape (Nx, Ny, 2) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=2]
        2D array of likelihood values corresponding to `abscissa`.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs2d_hdf5(abscissa,
                    likelihood,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)


cpdef brejs3d_hdf5(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] likelihood,
        object dataset,
        unsigned long chunk_size = 10_000,
        unsigned long max_iterations = 10_000,
        bint show_progress = False):
    """
    Python wrapper for 3D proposal rejection sampling.

    This function exposes the C-level `_prejs3d` function to Python. It generates samples from a target
    3D likelihood function using a non-uniform proposal distribution with inverse sampling on the specified axis.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        4D array of shape (Nx, Ny, Nz, 3) representing grid coordinates for the target likelihood.
    likelihood : np.ndarray[DTYPE_t, ndim=3]
        3D array of likelihood values corresponding to `abscissa`.
    dataset: HDF5 Dataset
        The dataset into which to deposit the data.
    chunk_size : unsigned long, optional
        Number of proposals generated per iteration (default: 10,000).
    max_iterations : unsigned long, optional
        Maximum number of iterations to perform (default: 10,000).
    show_progress : bint
        whether to display a tqdm progress bar in Python.
    Returns
    -------
    None
        Samples are stored in `result_buffer`.
    """
    return _brejs3d_hdf5(abscissa,
                    likelihood,
                    dataset,
                    chunk_size=chunk_size,
                    max_iterations=max_iterations,
                    show_progress=show_progress)
