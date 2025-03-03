# cython: boundscheck=False, wraparound=False, cdivision=True, profile=True

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
import numpy as pnp

cimport numpy as np

from pisces.utilities.math_utils cimport _linterp


# Setup the random number generating functions.
cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil
srand48(-100)

cdef void _invcsamp_cdf(
    DTYPE_t[: ] x,
    DTYPE_t[: ] cdf,
    DTYPE_t[: ] results_buffer
):
    """
    Perform inverse-transform sampling given an already normalized CDF.

    * x and cdf should have length n.
      - cdf must be strictly increasing and end with cdf[-1] = 1.0
    * results_buffer is where we store the sample positions (also length m),
      which can differ from n.

    Steps:
      1. Fill `results_buffer` with uniform random values in [0,1].
      2. Use linear interpolation (`_linterp1d`) to map each uniform
         random to the corresponding x-value where cdf = uniform.
    """
    # Create the random uniform sample of points from which
    # we will assign position values.
    cdef Py_ssize_t nsamples = results_buffer.shape[0], i
    for i in range(nsamples):
        results_buffer[i] = drand48()

    # Pass the buffer to the interpolation system so that
    # we can generate the random sample.
    _linterp._linterp1d(results_buffer,cdf,x,results_buffer)

    return

cpdef np.ndarray[DTYPE_t, ndim=1] invcsamp_cdf(
        np.ndarray[DTYPE_t, ndim=1] x,
        np.ndarray[DTYPE_t, ndim=1] cdf,
        unsigned long long n):
    # Declare the results buffer.
    cdef np.ndarray[DTYPE_t, ndim=1] results_buffer = pnp.empty((n,),dtype='f8')

    # Create memory views to allow fast interaction:
    cdef DTYPE_t[: ] xmv = x, cdfmv = cdf , rbmv = results_buffer

    # pass the buffer down
    _invcsamp_cdf(xmv,cdfmv,rbmv)

    return results_buffer
