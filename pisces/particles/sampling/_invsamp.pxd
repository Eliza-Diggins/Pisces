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
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.uint64_t INT_TYPE_t
ctypedef np.intp_t NATINT_TYPE_t

cdef void _invcsamp_cdf(
    DTYPE_t[: ] x,
    DTYPE_t[: ] cdf,
    DTYPE_t[: ] results_buffer
)

cpdef invcsamp_cdf(
        np.ndarray[DTYPE_t, ndim=1] x,
        np.ndarray[DTYPE_t, ndim=1] cdf,
        unsigned long long n)
