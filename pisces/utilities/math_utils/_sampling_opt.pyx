# cython: boundscheck=False, wraparound=False, cdivision=True

# ============================================================ #
# Optimized particle sampling functions                        #
#                                                              #
# The functions in this module provide C-level functions for   #
# sampling particles from specific density profiles. These     #
# should never be called directly as they have no handling     #
# for Jacobian scaling etc.                                    #
#                                                              #
# Written by: Eliza Diggins, copyright 2025                    #
# ============================================================ #
import numpy as np
cimport numpy as np
from tqdm.auto import tqdm

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t INT_TYPE_t

# @@ UTILITY ROUTINES @@ #
# These routines are simple utility functions called elsewhere in the codebase.
cdef inline np.ndarray[DTYPE_t, ndim=1] trapezoid_integrate(
    np.ndarray[DTYPE_t, ndim=1] y,
    np.ndarray[DTYPE_t, ndim=1] x
):
    """
    Perform trapezoidal integration of an array.

    Parameters
    ----------
    y : np.ndarray[DTYPE_t, ndim=1]
        The values of the function to integrate.

    x : np.ndarray[DTYPE_t, ndim=1]
        The abscissa (x-values) corresponding to `y`.

    Returns
    -------
    DTYPE_t
        The result of the trapezoidal integration.
    """
    # Define the result array that is filled by the result of the integration.
    cdef np.ndarray[DTYPE_t, ndim = 1] result = np.zeros((len(x),),dtype=np.float64)

    # Iterate through each of the relevant abscissa points and perform
    # the quadrature.
    for i in range(1, len(x)):
        result[i] = result[i-1] + (0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]))

    return result

# @@ LINEAR INTERPOLATION @@ #
# The linear interpolation schemes here are used in the rejection sampling methods to
# determine density values between abscissa points.
cdef inline np.ndarray[DTYPE_t, ndim=1] interp1d_base(
        np.ndarray[DTYPE_t, ndim=1] x,
        np.ndarray[DTYPE_t, ndim=1] x0,
        np.ndarray[DTYPE_t, ndim=1] x1,
        np.ndarray[DTYPE_t, ndim=1] y0,
        np.ndarray[DTYPE_t, ndim=1] y1):
    """
    Compute the interpolated value at ``x`` given the two bordering
    ``x`` and ``y`` values. This method is valid for vectorized inputs.
    
    Parameters
    ----------
    x: np.ndarray[float64]
        The points at which to perform the interpolation.
    x0: np.ndarray[float64]
        The left abscissa points.
    x1: np.ndarray[float64]
        The right abscissa points.
    y0: np.ndarray[float64]
        The left hand value of the interpolant.
    y1: np.ndarray[float64]
        The right hand value of the interpolant.

    Returns
    -------
    np.ndarray[float64]
    The resulting interpolated values.
    """
    return y0 + ((x-x0)*(y1-y0)/(x1-x0))

cdef inline np.ndarray[DTYPE_t, ndim=1] interp1d(
        np.ndarray[DTYPE_t, ndim=1] inputs,
        np.ndarray[DTYPE_t, ndim=1] x,
        np.ndarray[DTYPE_t, ndim=1] y):
    """
    Construct and evaluate the 1-D linear interpolation over a specific abscissa and
    interpolant.
    
    Parameters
    ----------
    inputs: np.ndarray[float64]
        The points in the domain of the absicssa where the function is to be
        evaluated.
    x: np.ndarray[float64]
        The absicssa on which to compute the interpolation.
    y: np.ndarray[float64]
        The values corresponding to each ``x`` value.

    Returns
    -------
    np.ndarray[float64]
        The resulting interpolation values.
    """
    # Declare internal variables
    cdef np.ndarray[DTYPE_t, ndim=1] x0, x1, y0, y1
    cdef np.ndarray[np.int64_t, ndim=1] idx

    # Create the indices.
    idx = np.searchsorted(x, inputs)
    idx = np.clip(idx, 1, len(x)-1)

    # Get the interpolation stencil
    x0,x1 = x[idx-1], x[idx]
    y0,y1 = y[idx-1], y[idx]

    # return
    return y0 + ((inputs-x0)*(y1-y0)/(x1-x0))



cdef np.ndarray[DTYPE_t, ndim=1] interp2d(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] interpolation_values,
        np.ndarray[DTYPE_t, ndim=2] inputs):
    """
    Perform linear interpolation in 2 dimensions.
    
    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        The grid of points for interpolation.
    
    interpolation_values : np.ndarray[DTYPE_t, ndim=2]
        Values defined on the grid.
    
    inputs : np.ndarray[DTYPE_t, ndim=2]
        Query points for interpolation.
    
    Returns
    -------
    np.ndarray[DTYPE_t, ndim=1]
        Interpolated values at the query points.
    """
    # Declare internal variables
    cdef np.ndarray[np.int64_t, ndim=1] index_x, index_y  # Indices for defining the stencil
    cdef np.ndarray[DTYPE_t, ndim=1] I00, I01, I10, I11  # Interpolation values in the stencil
    cdef np.ndarray[DTYPE_t, ndim=1] x0, x1, y0, y1  # Stencil points
    cdef np.ndarray[DTYPE_t, ndim=1] Q0, Q1, Z  # Intermediate and final results
    cdef int n = inputs.shape[0]  # Number of query points
    # Identify indices for the query points in the grid
    index_x = np.searchsorted(abscissa[:, 0, 0], inputs[:, 0])
    index_y = np.searchsorted(abscissa[0, :, 1], inputs[:, 1])

    # Ensure indices stay within valid bounds
    index_x = np.clip(index_x, 1, abscissa.shape[0] - 1)
    index_y = np.clip(index_y, 1, abscissa.shape[1] - 1)

    # Pull interpolation values
    I00 = interpolation_values[index_x - 1, index_y - 1]
    I01 = interpolation_values[index_x - 1, index_y]
    I10 = interpolation_values[index_x, index_y - 1]
    I11 = interpolation_values[index_x, index_y]

    # Pull abscissa points from the grid
    x0 = abscissa[index_x - 1, 0, 0]
    x1 = abscissa[index_x, 0, 0]
    y0 = abscissa[0, index_y - 1, 1]
    y1 = abscissa[0, index_y, 1]

    # Interpolate along x-axis
    Q0 = interp1d_base(inputs[:, 0], x0, x1, I00, I10)
    Q1 = interp1d_base(inputs[:, 0], x0, x1, I01, I11)

    # Interpolate along y-axis
    Z = interp1d_base(inputs[:, 1], y0, y1, Q0, Q1)

    return Z

cdef np.ndarray[DTYPE_t, ndim=1] interp3d(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] interpolation_values,
        np.ndarray[DTYPE_t, ndim=2] inputs):
    """
    Perform linear interpolation in 3 dimensions.
    
    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        The grid of points for interpolation.
    
    interpolation_values : np.ndarray[DTYPE_t, ndim=3]
        Values defined on the grid.
    
    inputs : np.ndarray[DTYPE_t, ndim=2]
        Query points for interpolation.
    
    Returns
    -------
    np.ndarray[DTYPE_t, ndim=1]
        Interpolated values at the query points.
    """
    # Declare internal variables
    cdef np.ndarray[np.int64_t, ndim=1] index_x, index_y, index_z  # Indices for defining the stencil
    cdef np.ndarray[DTYPE_t, ndim=1] x0, x1, y0, y1, z0, z1  # Stencil points
    cdef np.ndarray[DTYPE_t, ndim=1] Q00, Q01, Q10, Q11, Q0, Q1, Z  # Intermediate and final results
    cdef np.ndarray[DTYPE_t, ndim=1] I000, I001, I010, I011, I100, I101, I110, I111  # Interpolation values in the stencil
    cdef int n = inputs.shape[0]  # Number of query points

    # Identify indices for the query points in the grid
    index_x = np.searchsorted(abscissa[:, 0, 0, 0], inputs[:, 0])
    index_y = np.searchsorted(abscissa[0, :, 0, 1], inputs[:, 1])
    index_z = np.searchsorted(abscissa[0, 0, :, 2], inputs[:, 2])

    # Ensure indices stay within valid bounds
    index_x = np.clip(index_x, 1, abscissa.shape[0] - 1)
    index_y = np.clip(index_y, 1, abscissa.shape[1] - 1)
    index_z = np.clip(index_z, 1, abscissa.shape[2] - 1)

    # Pull interpolation values
    I000 = interpolation_values[index_x - 1, index_y - 1, index_z - 1]
    I001 = interpolation_values[index_x - 1, index_y - 1, index_z]
    I010 = interpolation_values[index_x - 1, index_y, index_z - 1]
    I011 = interpolation_values[index_x - 1, index_y, index_z]
    I100 = interpolation_values[index_x, index_y - 1, index_z - 1]
    I101 = interpolation_values[index_x, index_y - 1, index_z]
    I110 = interpolation_values[index_x, index_y, index_z - 1]
    I111 = interpolation_values[index_x, index_y, index_z]

    # Pull abscissa points from the grid
    x0 = abscissa[index_x - 1, 0, 0, 0]
    x1 = abscissa[index_x, 0, 0, 0]
    y0 = abscissa[0, index_y - 1, 0, 1]
    y1 = abscissa[0, index_y, 0, 1]
    z0 = abscissa[0, 0, index_z - 1, 2]
    z1 = abscissa[0, 0, index_z, 2]

    # Interpolate along x-axis
    Q00 = interp1d_base(inputs[:, 0], x0, x1, I000, I100)
    Q01 = interp1d_base(inputs[:, 0], x0, x1, I001, I101)
    Q10 = interp1d_base(inputs[:, 0], x0, x1, I010, I110)
    Q11 = interp1d_base(inputs[:, 0], x0, x1, I011, I111)

    # Interpolate along y-axis
    Q0 = interp1d_base(inputs[:, 1], y0, y1, Q00, Q10)
    Q1 = interp1d_base(inputs[:, 1], y0, y1, Q01, Q11)

    # Interpolate along z-axis
    Z = interp1d_base(inputs[:, 2], z0, z1, Q0, Q1)

    return Z

# REJECTION SAMPLING ROUTINES
# These functions implement rejection sampling for the specified
# number of dimensions being considered.
def rejection_sampling_2D(
    np.ndarray[DTYPE_t, ndim=3] abscissa,
    np.ndarray[DTYPE_t, ndim=2] field,
    unsigned long long count,
    unsigned long long chunk_size,
    np.ndarray[DTYPE_t, ndim=2] result_buffer,
    unsigned long long max_iterations,
    int progress_bar_flag,
):
    """
    Sample a set of particle positions from a generic PDF function evaluated on a specific absicssa grid.
    This implementation provides only a uniform distribution for sampling.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        The 2D grid of points, shape (N1, N2, 2), where the last dimension represents (x, y).

    field : np.ndarray[DTYPE_t, ndim=2]
        The probability density function (PDF) values defined on the grid, shape (N1, N2).

    count : unsigned long long
        The number of samples to generate.

    chunk_size : unsigned long long
        The size of each chunk of samples to generate in one iteration.

    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        A pre-allocated buffer to store the sampled points, shape (count, 2).

    max_iterations : unsigned long long
        The maximum number of iterations to attempt sampling.

    progress_bar_flag: int
        Indicator to show whether or not to disable the progress bar.

    Returns
    -------
    np.ndarray[DTYPE_t, ndim=2]
        The sampled points, shape (count, 2).
    """
    # --------------- VARIABLE DECLARATION ---------------- #
    # FLAGS: signaling and tracking variables.
    #   n_accept: The number of particles accepted.
    #   iterations: The number of iterations.
    #   i: The flag for the current iteration.
    #   _abs_indx_0, _abs_indx_1: Flags indicating the edge indices of the abscissa.
    #   accepted_indices: Integer array containing the set of indices to accept at each cycle.
    cdef unsigned long long n_accept = 0
    cdef unsigned long long iterations = 0
    cdef unsigned long long i, _abs_indx_0, _abs_indx_1
    cdef np.ndarray[INT_TYPE_t, ndim=1] accepted_indices

    # CORE ARRAYS: These are the arrays that handle the core of the computation.
    #   random_points: stores the sampled points in the abscissa.
    #   uniform_values: stores the random uniform values used for checking acceptance.
    #   interpolated_values: The values of the PDF at the random points.
    cdef np.ndarray[DTYPE_t, ndim=2] random_points = np.empty((chunk_size, 2), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] uniform_values = np.empty(chunk_size, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] interpolated_values = np.empty((chunk_size,),dtype=np.float64)

    # GENERIC FLOATS: Used for storing various basic variable measures.
    cdef DTYPE_t x, y, max_pdf, x_min,x_max,y_min,y_max, min_pdf

    # -------------- SETUP PROCEDURES ---------------- #
    # In this section of the code, we determine the
    # boundaries on the abscissa and then rescale
    # the PDF to be valid. We also setup the progress
    # bar object that will be used.

    # Identify the abscissa boundaries. Cython doesn't permit negative
    # indices so we need to identify the length of the abscissa in each direction.
    _abs_indx_0,_abs_indx_1 = abscissa.shape[0]-1,abscissa.shape[1]-1
    x_min, x_max = abscissa[0, 0, 0],abscissa[_abs_indx_0, 0, 0]
    y_min, y_max = abscissa[0, 0, 1], abscissa[0, _abs_indx_1, 1]

    # Rescale the PDF provided.
    # We need the field to exist between 0 and 1. We place
    # the maximum at 1 so that we have the best acceptance ratio.
    # If there are values below zero, an error is raised.
    max_pdf,min_pdf = np.amax(field),np.amin(field)
    if min_pdf < 0:
        raise ValueError("PDF had values < 0.")
    field /= max_pdf

    # Setup the progress bar.
    pbar = tqdm(leave=True,
                total=count,
                desc="Rejection Sampling [2D, r=0, Iter=0]",
                disable=bool(progress_bar_flag)
                )

    # --------------- MAIN LOOP -------------------- #
    # For each iteration, we perform a chunk's worth of
    # samples and check them. For each accepted value, we add
    # it to the selected answers list and proceed.
    while n_accept < count and iterations < max_iterations:
        # Setup this iteration. Increment flags as necessary and
        # sample the randomized points.
        iterations += 1

        # Sample the points.
        random_points[:, 0] = np.random.uniform(x_min, x_max, chunk_size)
        random_points[:, 1] = np.random.uniform(y_min, y_max, chunk_size)
        uniform_values[:] = np.random.uniform(0, max_pdf, chunk_size)
        interpolated_values = interp2d(abscissa,field,random_points)

        # Determine acceptances and rejections. Count the successes and add them
        # to the result buffer.
        accepted_indices = np.nonzero(uniform_values <= interpolated_values)[0]
        accepted_count = len(accepted_indices)

        if n_accept + accepted_count > count:
            # Adjust the number of accepted points to fill the buffer exactly
            accepted_count = count - n_accept

        # Store the accepted points in the result buffer
        result_buffer[n_accept:n_accept + accepted_count, :] = random_points[accepted_indices][:accepted_count]

        # Finish the iteration by incrementing the total successes and
        # updating the progress bar.
        n_accept += accepted_count
        pbar.update(accepted_count)
        pbar.desc = "Rejection Sampling [2D, r=%s, Iter=%s]"%(np.format_float_scientific(float(n_accept)/float(iterations*chunk_size),precision=6),iterations)
    pbar.close()

    # Check if we failed to generate enough samples
    if n_accept < count:
        raise RuntimeError("Rejection sampling failed to generate enough samples within max_iterations.")

    return result_buffer

def rejection_sampling_2D_proposal(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] field,
        unsigned long long count,
        unsigned long long chunk_size,
        np.ndarray[DTYPE_t, ndim=1] proposal_field,
        int proposal_axis,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long long max_iterations,
        int progress_bar_flag
):
    """

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        The 2D grid of points, shape (N1, N2, 2), where the last dimension represents (x, y).

    field : np.ndarray[DTYPE_t, ndim=3]
        The probability density function (PDF) values defined on the grid, shape (N1, N2).

    count : unsigned long long
        The number of samples to generate.

    chunk_size : unsigned long long
        The size of each chunk of samples to generate in one iteration.
    proposal_field: np.ndarray[DTYPE_t, ndim=1]
        The 1D proposal field to sample from.
    proposal_axis: int
        The index of the proposal axis.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        A pre-allocated buffer to store the sampled points, shape (count, 2).

    max_iterations : unsigned long long
        The maximum number of iterations to attempt sampling.

    progress_bar_flag: int
        Indicator to show whether or not to disable the progress bar.

    Returns
    -------
    np.ndarray[DTYPE_t, ndim=2]
        The sampled points, shape (count, 2).

    """
    # -------------- VARIABLE DECLARATION ----------------- #
    # abscissa related variables:
    #   - _abscissa_proposal: The abscissa for the proposal PDF.
    #   - x_min, x_max, y_min, y_max: The limits of the abscissa.
    #   - _abs_prop_min, _abs_prop_max: The min and max of the abscissa for the proposal.
    #   - free_axis_index indicates the axis which is not dedicated to the proposal.
    #   - _abs_free_min, _abs_free_max: The min and max of the abscissa for the proposal.
    cdef np.ndarray[DTYPE_t, ndim=1] _abscissa_proposal
    cdef DTYPE_t x_min,x_max,y_min,y_max,_abs_prop_min,_abs_prop_max, _abs_free_min,_abs_free_max
    cdef int free_axis_index

    # Scaling variables:
    #   - scale_factor: The ratio between the maximum pdf and maximum pdf proposal.
    cdef DTYPE_t scale_factor

    # Sampling variables:
    #   - proposal_cdf: The CDF array for the proposal.
    #   - random_points: The random sample points from the abscissa.
    #   - uniform_values: The uniform reference values
    #   - interpolated_values: The interpolated field values at the random points.
    cdef np.ndarray[DTYPE_t, ndim=1] proposal_cdf
    cdef np.ndarray[DTYPE_t, ndim=2] random_points = np.empty((chunk_size, 2), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] uniform_values = np.empty(chunk_size, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] interpolated_values = np.empty((chunk_size,2), dtype=np.float64)

    # Iteration Flags:
    cdef unsigned long long n_accept = 0
    cdef unsigned long long iterations = 0
    cdef np.ndarray[INT_TYPE_t, ndim=1] accepted_indices
    cdef unsigned long long i, _abs_indx_0, _abs_indx_1,


    # -------------- SETUP PROCEDURES ---------------- #
    # In this section of the code, we determine the
    # boundaries on the abscissa and then rescale
    # the PDF to be valid. We also setup the progress
    # bar object that will be used.
    #
    # Because we are now using a powerlaw proposal function,
    # more effort goes into rescaling and then setting
    # ourselves up to interpolate samples from the
    # inverse CDF.

    # Identify the abscissa boundaries. Using the proposal index, identify and
    # isolate the abscissa for the proposal.
    _abs_indx_0, _abs_indx_1 = abscissa.shape[0] - 1, abscissa.shape[1] - 1
    x_min, x_max = abscissa[0, 0, 0], abscissa[_abs_indx_0, 0, 0]
    y_min, y_max = abscissa[0, 0, 1], abscissa[0, _abs_indx_1, 1]

    if proposal_axis == 0:
        _abscissa_proposal = abscissa[:,0,0]
        _abs_prop_min, _abs_prop_max = x_min,x_max
        _abs_free_min,_abs_free_max = y_min,y_max
        free_axis_index = 1
    else:
        _abscissa_proposal = abscissa[0,:,1]
        _abs_prop_min, _abs_prop_max = y_min, y_max
        _abs_free_min,_abs_free_max = x_min,x_max
        free_axis_index = 0


    # Rescale the PDF. In this case, we don't need to have a particular scaling,
    # we just need to ensure that both PDFs are larger than zero and that
    # we can find a rescaling.
    if np.amin(field) < 0:
        raise ValueError("PDF had values < 0.")
    if np.amin(field) < 0:
        raise ValueError("Proposal PDF had values < 0.")

    if proposal_axis == 0:
        scale_factor = np.amax(np.amax(field,axis=1)/proposal_field)
    else:
        scale_factor = np.amax(np.amax(field, axis=0) / proposal_field)

    proposal_field *= scale_factor

    # -------------- SAMPLING SETUP ----------------- #
    # In order to setup the sampling, we need to compute a CDF for this
    # proposal field in the native coordinates. We'll treat it as linear
    # between samples and then integrate (trapazoid rule).
    proposal_cdf = trapezoid_integrate(proposal_field, _abscissa_proposal)
    proposal_cdf /= np.amax(proposal_cdf)

    # Setup the progress bar.
    pbar = tqdm(leave=True,
                total=count,
                desc="Rejection Sampling [2D, r=0, Iter=0]",
                disable=bool(progress_bar_flag)
                )

    # --------------- MAIN LOOP -------------------- #
    # For each iteration, we perform a chunk's worth of
    # samples and check them. For each accepted value, we add
    # it to the selected answers list and proceed.
    while n_accept < count and iterations < max_iterations:
        # Setup this iteration. Increment flags as necessary and
        # sample the randomized points.
        iterations += 1

        # Sample the points. We pull uniforms into the free axis index and then
        # we pass the uniforms into the CDF to sample the proposal axis.
        random_points[:,free_axis_index] = np.random.uniform(_abs_free_min,_abs_free_max, chunk_size)
        random_points[:,proposal_axis] = interp1d(np.random.uniform(0, 1, chunk_size), proposal_cdf, _abscissa_proposal)
        uniform_values = np.random.uniform(0,1,chunk_size)

        # Interpolate into the field and the proposal_field
        interpolated_values[:,0] = interp2d(abscissa, field, random_points)
        interpolated_values[:,1] = interp1d(random_points[:, proposal_axis], _abscissa_proposal, proposal_field)

        # Determine acceptances and rejections. Count the successes and add them
        # to the result buffer.
        accepted_indices = np.nonzero(uniform_values <= interpolated_values[:,0]/interpolated_values[:,1])[0]
        accepted_count = len(accepted_indices)

        if n_accept + accepted_count > count:
            # Adjust the number of accepted points to fill the buffer exactly
            accepted_count = count - n_accept

        # Store the accepted points in the result buffer
        result_buffer[n_accept:n_accept + accepted_count, :] = random_points[accepted_indices,:][:accepted_count,:]

        # Finish the iteration by incrementing the total successes and
        # updating the progress bar.
        n_accept += accepted_count
        pbar.update(accepted_count)
        pbar.desc = "Rejection Sampling [2D, r=%s, Iter=%s]"%(np.format_float_scientific(float(n_accept)/float(iterations*chunk_size),precision=6),iterations)
    pbar.close()

    # Check if we failed to generate enough samples
    if n_accept < count:
        raise RuntimeError("Rejection sampling failed to generate enough samples within max_iterations.")

    return result_buffer

def rejection_sampling_3D(
        np.ndarray[DTYPE_t, ndim=4] abscissa,
        np.ndarray[DTYPE_t, ndim=3] field,
        unsigned long long count,
        unsigned long long chunk_size,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long long max_iterations,
        int progress_bar_flag,
):
    """
    Sample a set of particle positions from a generic PDF function evaluated on a specific absicssa grid.
    This implementation provides only a uniform distribution for sampling.

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=4]
        The 2D grid of points, shape (N1, N2, N3, 3), where the last dimension represents (x, y, z).

    field : np.ndarray[DTYPE_t, ndim=2]
        The probability density function (PDF) values defined on the grid, shape (N1, N2, N3).

    count : unsigned long long
        The number of samples to generate.

    chunk_size : unsigned long long
        The size of each chunk of samples to generate in one iteration.

    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        A pre-allocated buffer to store the sampled points, shape (count, 3).

    max_iterations : unsigned long long
        The maximum number of iterations to attempt sampling.

    progress_bar_flag: int
        Indicator to show whether or not to disable the progress bar.

    Returns
    -------
    np.ndarray[DTYPE_t, ndim=3]
        The sampled points, shape (count, 3).
    """
    # --------------- VARIABLE DECLARATION ---------------- #
    # FLAGS: signaling and tracking variables.
    #   n_accept: The number of particles accepted.
    #   iterations: The number of iterations.
    #   i: The flag for the current iteration.
    #   _abs_indx_0, _abs_indx_1: Flags indicating the edge indices of the abscissa.
    #   accepted_indices: Integer array containing the set of indices to accept at each cycle.
    cdef unsigned long long n_accept = 0
    cdef unsigned long long iterations = 0
    cdef unsigned long long i, _abs_indx_0, _abs_indx_1, _abs_indx_2
    cdef np.ndarray[INT_TYPE_t, ndim=1] accepted_indices

    # CORE ARRAYS: These are the arrays that handle the core of the computation.
    #   random_points: stores the sampled points in the abscissa.
    #   uniform_values: stores the random uniform values used for checking acceptance.
    #   interpolated_values: The values of the PDF at the random points.
    cdef np.ndarray[DTYPE_t, ndim=2] random_points = np.empty((chunk_size, 3), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] uniform_values = np.empty(chunk_size, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] interpolated_values = np.empty((chunk_size,), dtype=np.float64)

    # GENERIC FLOATS: Used for storing various basic variable measures.
    cdef DTYPE_t x, y, max_pdf, x_min, x_max, y_min, y_max, z_min,z_max, min_pdf

    # -------------- SETUP PROCEDURES ---------------- #
    # In this section of the code, we determine the
    # boundaries on the abscissa and then rescale
    # the PDF to be valid. We also setup the progress
    # bar object that will be used.

    # Identify the abscissa boundaries. Cython doesn't permit negative
    # indices so we need to identify the length of the abscissa in each direction.
    _abs_indx_0, _abs_indx_1, _abs_indx_2 = abscissa.shape[0] - 1, abscissa.shape[1] - 1, abscissa.shape[2]-1
    x_min, x_max = abscissa[0, 0, 0, 0], abscissa[_abs_indx_0, 0, 0, 0]
    y_min, y_max = abscissa[0, 0, 0, 1], abscissa[0, _abs_indx_1, 0, 1]
    z_min, z_max = abscissa[0, 0, 0, 2], abscissa[0, 0, _abs_indx_2, 2]

    # Rescale the PDF provided.
    # We need the field to exist between 0 and 1. We place
    # the maximum at 1 so that we have the best acceptance ratio.
    # If there are values below zero, an error is raised.
    max_pdf, min_pdf = np.amax(field), np.amin(field)
    if min_pdf < 0:
        raise ValueError("PDF had values < 0.")
    field /= max_pdf

    # Setup the progress bar.
    pbar = tqdm(leave=True,
                total=count,
                desc="Rejection Sampling [3D, r=0, Iter=0]",
                disable=bool(progress_bar_flag)
                )

    # --------------- MAIN LOOP -------------------- #
    # For each iteration, we perform a chunk's worth of
    # samples and check them. For each accepted value, we add
    # it to the selected answers list and proceed.
    while n_accept < count and iterations < max_iterations:
        # Setup this iteration. Increment flags as necessary and
        # sample the randomized points.
        iterations += 1

        # Sample the points.
        random_points[:, 0] = np.random.uniform(x_min, x_max, chunk_size)
        random_points[:, 1] = np.random.uniform(y_min, y_max, chunk_size)
        random_points[:, 2] = np.random.uniform(z_min,z_max, chunk_size)
        uniform_values[:] = np.random.uniform(0, max_pdf, chunk_size)
        interpolated_values = interp3d(abscissa, field, random_points)

        # Determine acceptances and rejections. Count the successes and add them
        # to the result buffer.
        accepted_indices = np.nonzero(uniform_values <= interpolated_values)[0]
        accepted_count = len(accepted_indices)

        if n_accept + accepted_count > count:
            # Adjust the number of accepted points to fill the buffer exactly
            accepted_count = count - n_accept

        # Store the accepted points in the result buffer
        result_buffer[n_accept:n_accept + accepted_count, :] = random_points[accepted_indices][:accepted_count]

        # Finish the iteration by incrementing the total successes and
        # updating the progress bar.
        n_accept += accepted_count
        pbar.update(accepted_count)
        pbar.desc = "Rejection Sampling [3D, r=%s, Iter=%s]" % (
        np.format_float_scientific(float(n_accept) / float(iterations * chunk_size), precision=6), iterations)
    pbar.close()

    # Check if we failed to generate enough samples
    if n_accept < count:
        raise RuntimeError("Rejection sampling failed to generate enough samples within max_iterations.")

    return result_buffer

def rejection_sampling_3D_proposal(
        np.ndarray[DTYPE_t, ndim=3] abscissa,
        np.ndarray[DTYPE_t, ndim=2] field,
        unsigned long long count,
        unsigned long long chunk_size,
        np.ndarray[DTYPE_t, ndim=1] proposal_field,
        int proposal_axis,
        np.ndarray[DTYPE_t, ndim=2] result_buffer,
        unsigned long long max_iterations,
        int progress_bar_flag
):
    """

    Parameters
    ----------
    abscissa : np.ndarray[DTYPE_t, ndim=3]
        The 2D grid of points, shape (N1, N2, 2), where the last dimension represents (x, y).

    field : np.ndarray[DTYPE_t, ndim=2]
        The probability density function (PDF) values defined on the grid, shape (N1, N2).

    count : unsigned long long
        The number of samples to generate.

    chunk_size : unsigned long long
        The size of each chunk of samples to generate in one iteration.
    proposal_field: np.ndarray[DTYPE_t, ndim=1]
        The 1D proposal field to sample from.
    proposal_axis: np.ndarray[DTYPE_t, ndim=1]
        The index of the proposal axis.
    result_buffer : np.ndarray[DTYPE_t, ndim=2]
        A pre-allocated buffer to store the sampled points, shape (count, 2).

    max_iterations : unsigned long long
        The maximum number of iterations to attempt sampling.

    progress_bar_flag: int
        Indicator to show whether or not to disable the progress bar.

    Returns
    -------
    np.ndarray[DTYPE_t, ndim=2]
        The sampled points, shape (count, 2).

    """
    # -------------- VARIABLE DECLARATION ----------------- #
    # abscissa related variables:
    #   - _abscissa_proposal: The abscissa for the proposal PDF.
    #   - x_min, x_max, y_min, y_max: The limits of the abscissa.
    #   - _abs_prop_min, _abs_prop_max: The min and max of the abscissa for the proposal.
    #   - free_axis_index indicates the axis which is not dedicated to the proposal.
    #   - _abs_free_min, _abs_free_max: The min and max of the abscissa for the proposal.
    cdef np.ndarray[DTYPE_t, ndim=1] _abscissa_proposal
    cdef DTYPE_t x_min, x_max, y_min, y_max, _abs_prop_min, _abs_prop_max, _abs_free2_min, _abs_free2_max, _abs_free1_min, _abs_free1_max, z_min,z_max
    cdef int free_axis_index_1, free_axis_index_2

    # Scaling variables:
    #   - scale_factor: The ratio between the maximum pdf and maximum pdf proposal.
    cdef DTYPE_t scale_factor

    # Sampling variables:
    #   - proposal_cdf: The CDF array for the proposal.
    #   - random_points: The random sample points from the abscissa.
    #   - uniform_values: The uniform reference values
    #   - interpolated_values: The interpolated field values at the random points.
    cdef np.ndarray[DTYPE_t, ndim=1] proposal_cdf
    cdef np.ndarray[DTYPE_t, ndim=2] random_points = np.empty((chunk_size, 2), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] uniform_values = np.empty(chunk_size, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=2] interpolated_values = np.empty((chunk_size, 2), dtype=np.float64)

    # Iteration Flags:
    cdef unsigned long long n_accept = 0
    cdef unsigned long long iterations = 0
    cdef np.ndarray[INT_TYPE_t, ndim=1] accepted_indices
    cdef unsigned long long i, _abs_indx_0, _abs_indx_1, _abs_indx_2

    # -------------- SETUP PROCEDURES ---------------- #
    # In this section of the code, we determine the
    # boundaries on the abscissa and then rescale
    # the PDF to be valid. We also setup the progress
    # bar object that will be used.
    #
    # Because we are now using a powerlaw proposal function,
    # more effort goes into rescaling and then setting
    # ourselves up to interpolate samples from the
    # inverse CDF.

    # Identify the abscissa boundaries. Using the proposal index, identify and
    # isolate the abscissa for the proposal.
    _abs_indx_0, _abs_indx_1, _abs_indx_2 = abscissa.shape[0] - 1, abscissa.shape[1] - 1, abscissa.shape[2]-1
    x_min, x_max = abscissa[0, 0, 0, 0], abscissa[_abs_indx_0, 0, 0, 0]
    y_min, y_max = abscissa[0, 0, 0, 1], abscissa[0, _abs_indx_1, 0, 1]
    z_min, z_max = abscissa[0, 0, 0, 2], abscissa[0, 0, _abs_indx_2, 2]

    if proposal_axis == 0:
        _abscissa_proposal = abscissa[:, 0, 0,0]
        _abs_prop_min, _abs_prop_max = x_min, x_max
        _abs_free1_min, _abs_free1_max = y_min, y_max
        _abs_free2_min, _abs_free2_max = z_min, z_max
        free_axis_index_1 = 1
        free_axis_index_2 = 2
    elif proposal_axis == 1:
        _abscissa_proposal = abscissa[0, :, 0,1]
        _abs_prop_min, _abs_prop_max = y_min, y_max
        _abs_free1_min, _abs_free1_max = x_min, x_max
        _abs_free2_min, _abs_free2_max = z_min, z_max
        free_axis_index_1 = 0
        free_axis_index_2 = 2
    else:
        _abscissa_proposal = abscissa[0, 0, :,2]
        _abs_prop_min, _abs_prop_max = z_min, z_max
        _abs_free1_min, _abs_free1_max = x_min, x_max
        _abs_free2_min, _abs_free2_max = y_min, y_max
        free_axis_index_1 = 0
        free_axis_index_2 = 1


    # Rescale the PDF. In this case, we don't need to have a particular scaling,
    # we just need to ensure that both PDFs are larger than zero and that
    # we can find a rescaling.
    if np.amin(field) < 0:
        raise ValueError("PDF had values < 0.")
    if np.amin(field) < 0:
        raise ValueError("Proposal PDF had values < 0.")

    if proposal_axis == 0:
        scale_factor = np.amax(np.amax(field, axis=(1,2)) / proposal_field)
    elif proposal_axis == 1:
        scale_factor = np.amax(np.amax(field, axis=(0,2)) / proposal_field)
    else:
        scale_factor = np.amax(np.amax(field, axis=(0,1)) / proposal_field)

    proposal_field *= scale_factor

    # -------------- SAMPLING SETUP ----------------- #
    # In order to setup the sampling, we need to compute a CDF for this
    # proposal field in the native coordinates. We'll treat it as linear
    # between samples and then integrate (trapazoid rule).
    proposal_cdf = trapezoid_integrate(proposal_field, _abscissa_proposal)
    proposal_cdf /= np.amax(proposal_cdf)

    # Setup the progress bar.
    pbar = tqdm(leave=True,
                total=count,
                desc="Rejection Sampling [3D, r=0, Iter=0]",
                disable=bool(progress_bar_flag)
                )

    # --------------- MAIN LOOP -------------------- #
    # For each iteration, we perform a chunk's worth of
    # samples and check them. For each accepted value, we add
    # it to the selected answers list and proceed.
    while n_accept < count and iterations < max_iterations:
        # Setup this iteration. Increment flags as necessary and
        # sample the randomized points.
        iterations += 1

        # Sample the points. We pull uniforms into the free axis index and then
        # we pass the uniforms into the CDF to sample the proposal axis.
        random_points[:, free_axis_index_1] = np.random.uniform(_abs_free1_min,_abs_free1_max, chunk_size)
        random_points[:, free_axis_index_2] = np.random.uniform(_abs_free2_min, _abs_free2_max, chunk_size)
        random_points[:, proposal_axis] = interp1d(np.random.uniform(0, 1, chunk_size), proposal_cdf,
                                                   _abscissa_proposal)
        uniform_values = np.random.uniform(0, 1, chunk_size)

        # Interpolate into the field and the proposal_field
        interpolated_values[:, 0] = interp3d(abscissa, field, random_points)
        interpolated_values[:, 1] = interp1d(random_points[:, proposal_axis], _abscissa_proposal, proposal_field)

        # Determine acceptances and rejections. Count the successes and add them
        # to the result buffer.
        accepted_indices = np.nonzero(uniform_values <= interpolated_values[:, 0] / interpolated_values[:, 1])[0]
        accepted_count = len(accepted_indices)

        if n_accept + accepted_count > count:
            # Adjust the number of accepted points to fill the buffer exactly
            accepted_count = count - n_accept

        # Store the accepted points in the result buffer
        result_buffer[n_accept:n_accept + accepted_count, :] = random_points[accepted_indices, :][:accepted_count, :]

        # Finish the iteration by incrementing the total successes and
        # updating the progress bar.
        n_accept += accepted_count
        pbar.update(accepted_count)
        pbar.desc = "Rejection Sampling [3D, r=%s, Iter=%s]" % (
        np.format_float_scientific(float(n_accept) / float(iterations * chunk_size), precision=6), iterations)
    pbar.close()

    # Check if we failed to generate enough samples
    if n_accept < count:
        raise RuntimeError("Rejection sampling failed to generate enough samples within max_iterations.")

    return result_buffer

