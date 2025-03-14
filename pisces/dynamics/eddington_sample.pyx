#cython: language_level=3, boundscheck=False
"""
Cython functions for optimized sampling of particle velocities from an
ergodic / eddington distribution function.
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------
import numpy as np

cimport cython
cimport numpy as np

from scipy.interpolate import dfitpack
from tqdm.auto import tqdm

from libc.stdlib cimport free, malloc


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil
    double sin(double x) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t


cdef unsigned int _generate_velocities(
        np.ndarray[DTYPE_t, ndim=1] relpot,
        np.ndarray[DTYPE_t, ndim=1] velesc,
        np.ndarray[DTYPE_t, ndim=1] likelihood_max,
        DTYPE_t[:] velocity_result_buffer,
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] c,
        int k,
        bint show_progress = True,
        unsigned long max_tries = 100_000,
        ):
    """
    Generate particle velocities following an Eddington-like distribution function using
    an acceptance-rejection method specialized for spherical systems.

    This function repeatedly samples trial velocities up to `velesc[i]` (the escape velocity)
    for each particle. It then evaluates the combination f*v^2 and compares it against a
    random deviate scaled by `likelihood_max[i]`. The sampling loop exits successfully
    when `f*v^2 < random * likelihood_max[i]`. If the loop exceeds `max_tries`, the
    function returns an error code (1). On success for all particles, it returns 0.

    Parameters
    ----------
    relpot : np.ndarray of float, shape (N,)
        Array of relative potential values, one for each particle. Often denoted by ψ(r).
    velesc : np.ndarray of float, shape (N,)
        Array of escape velocities for each particle.
    likelihood_max : np.ndarray of float, shape (N,)
        Precomputed maximum values of f*v^2 for each particle. Used as the bounding function
        in the acceptance-rejection sampler.
    velocity_result_buffer : 1D buffer (memoryview or NumPy array), shape (N,)
        The output array into which sampled velocities will be written.
    t : np.ndarray of float
        Knot positions for the spline (required by `dfitpack.splev`).
    c : np.ndarray of float
        Spline coefficients for the distribution function (required by `dfitpack.splev`).
    k : int
        Spline degree for the distribution function.
    show_progress : bool, default True
        If True, a progress bar is shown (using `tqdm`).
    max_tries : unsigned long, default 100_000
        Maximum number of attempts to generate a valid velocity for each particle before
        returning an error code.

    Returns
    -------
    int
        A status code: returns 0 on success if all particle velocities are generated
        within `max_tries`, otherwise returns 1 if any particle exceeds `max_tries`.
    """
    # Declare variables: Core variables
    cdef Py_ssize_t num_particles = relpot.shape[0]

    # Declare variables: temporary placeholders
    cdef DTYPE_t v2, f
    cdef double * _e_ref = <double *> malloc(sizeof(double))
    cdef double[:] _e = <double[:1]> _e_ref

    # Declare variables: Flags
    cdef unsigned long ntries = 0
    cdef Py_ssize_t i = 0

    # Only declare the progress bar if we are asked to
    # declare one.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(leave=False,
                    total=num_particles,
                    desc="Generating particle velocities [Eddington]",
                    disable=~show_progress)

    # Perform the iteration step. We allow only MAX_TRIES chances to find a velocity before
    # returning a non-zero exit code and returning for a particular particle.
    for i in range(num_particles):
        # set the trial counter to 0
        ntries = 0

        # Perform the sampling loop.
        while ntries < max_tries:
            ntries += 1

            # Sample the velocity up to the escape velocity for
            # this particle.
            v2 = drand48()*velesc[i]
            v2 *= v2

            # Compute the relative energy and compute the value of the
            # distribution function for the relative energy.
            _e[0] = relpot[i]-0.5*v2
            f = dfitpack.splev(t,c, k, _e, 0)[0][0]

            # Check if we can exit.
            if f*v2 < drand48()*likelihood_max[i]:
                break

        # Check if we ran out of trials for this particle. If we did, the
        # error code needs to be returned.
        if ntries > max_tries:
            return 1

        # Write the velocity data to the result array.

        velocity_result_buffer[i] = sqrt(v2)
        if show_progress:
            pbar.update()

    # Close the progress bar.
    if show_progress:
        pbar.close()

    # Free memory
    free(_e_ref)

    # return
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_velocities(
    np.ndarray[DTYPE_t, ndim=1] relpot,
    np.ndarray[DTYPE_t, ndim=1] velesc,
    np.ndarray[DTYPE_t, ndim=1] likelihood_max,
    np.ndarray[DTYPE_t, ndim=1] t,
    np.ndarray[DTYPE_t, ndim=1] c,
    int k,
    bint show_progress=True,
    unsigned long max_tries=100000
):
    """
    Python-accessible wrapper around the Cython-level generate_velocities function.

    Parameters
    ----------
    relpot : np.ndarray of float, shape (N,)
        Relative potential for each particle.
    velesc : np.ndarray of float, shape (N,)
        Escape velocity for each particle.
    likelihood_max : np.ndarray of float, shape (N,)
        Maximum of f*v^2 for each particle.
    t : np.ndarray of float
        Knot positions for the spline (required by dfitpack.splev).
    c : np.ndarray of float
        Spline coefficients for the distribution function.
    k : int
        Spline degree for the distribution function.
    show_progress : bool, default True
        Whether to show a progress bar (tqdm).
    max_tries : unsigned long, default 100000
        Maximum number of attempts to sample a velocity for each particle
        before throwing an error.

    Returns
    -------
    velocity : np.ndarray of float, shape (N,)
        Generated velocities for each particle.

    Raises
    ------
    ValueError
        If the velocity generation fails for any particle within the
        allowed number of attempts.
    """
    cdef Py_ssize_t num_particles = relpot.shape[0]

    # Allocate a NumPy array for the velocity results.
    # We'll pass its data pointer into the lower-level function.
    cdef np.ndarray[DTYPE_t, ndim=1] velocity = np.zeros(num_particles, dtype=np.float64)

    # Call the lower-level function.
    cdef int status = _generate_velocities(
        relpot,
        velesc,
        likelihood_max,
        velocity,      # pass in velocity buffer
        t,
        c,
        k,
        show_progress,
        max_tries
    )

    # Interpret the status code:
    # 0 -> success; 1 -> too many tries
    if status != 0:
        raise ValueError(
            f"Failed to generate velocities for one or more particles "
            f"within {max_tries} attempts."
        )

    return velocity
