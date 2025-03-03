# cython: boundscheck=False, wraparound=False, cdivision=True
# ============================================================ #
# Pisces low-level linear interpolation implementations.       #
#                                                              #
# Written by: Eliza Diggins, University of Utah                #
# ============================================================ #
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.uint64_t INT_TYPE_t
ctypedef np.intp_t NATINT_TYPE_t

cdef void _linterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] result_buffer)

cdef void eval_clinterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] bounds,
        DTYPE_t[: ] result_buffer)

cdef void eval_elinterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] bounds,
        DTYPE_t[: ] result_buffer)

# ============================================================ #
# 2-D Linear interpolators                                     #
# ============================================================ #
cdef void _linterp2d(
        DTYPE_t[: , ::1] inputs,
        DTYPE_t[: , :, ::1] abscissa,
        DTYPE_t[: , ::1] ordinate,
        DTYPE_t[: ] result_buffer)

cdef void eval_clinterp2d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] ordinate,
        DTYPE_t[:] bounds,
        DTYPE_t[:] result_buffer)


cdef void eval_elinterp2d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] ordinate,
        DTYPE_t[:] bounds,
        DTYPE_t[:] result_buffer)

#
# # ============================================================ #
# # 3-D Linear interpolators                                     #
# # ============================================================ #
cdef void _linterp3d(
        DTYPE_t[: , ::1] inputs,
        DTYPE_t[: , :,:, ::1] abscissa,
        DTYPE_t[: ,:, ::1] ordinate,
        DTYPE_t[: ] result_buffer)

cdef void eval_clinterp3d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:, :, ::1] ordinate,
        DTYPE_t[:] bounds,
        DTYPE_t[:] result_buffer)

cdef void eval_elinterp3d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:, :, ::1] ordinate,
        DTYPE_t[:] bounds,
        DTYPE_t[:] result_buffer)
