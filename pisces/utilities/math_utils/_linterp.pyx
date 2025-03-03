# cython: boundscheck=False, wraparound=False, cdivision=True, profile=True
# ============================================================ #
# Pisces low-level linear interpolation implementations.       #
#                                                              #
# Written by: Eliza Diggins, University of Utah                #
# ============================================================ #
import numpy as pnp

cimport numpy as np

from Cython.Compiler.Future import absolute_import

from libc.stdlib cimport free, malloc


# ============================================================ #
# Utility Functions                                            #
# ============================================================ #
cdef inline unsigned long searchsorted(
    double[:] arr,
    unsigned long n,
    double val
) nogil:
    """
    'Left' variant of searchsorted for a sorted array `arr` of length `n`.
    Returns the insertion index as unsigned long.
    """
    cdef unsigned long left = 0
    cdef unsigned long right = n
    cdef unsigned long mid

    while left < right:
        mid = (left + right) >> 1  # (left + right) // 2, but bit-shift is often slightly faster
        if arr[mid] < val:
            left = mid + 1
        else:
            right = mid

    return left

# ============================================================ #
# 1-D Linear interpolators                                     #
# ============================================================ #
# All of these interpolation schemes are based on very generic
# linear interpolation. By default, each type is named as
# <action>_<boundary>linterp<ND>
cdef inline void _linxtnd(
         DTYPE_t[: ] x,
         DTYPE_t[:,::1] xs,
         DTYPE_t[:,::1] ys,
         DTYPE_t[: ] result_buffer):

    """
    Compute the value of the linear function passing through
    (x0,y0) and (x1,y1) at the abscissa point x.

    Parameters
    ----------
    x: float64 memory view
        The domain points to evaluate at. This should be an array of
        abscissa points.
    xs: float64 memory view
        The x points bounding each sample position. This should be (N,2) is
        shape with the first corresponding to the x value below and the second
        to the x value above the evaluation point.
    ys: float64 memory view
        The interpolation data for each point. Must be the same shape as ``xs``.
    result_buffer: float64 memory view
        This should be an ``(N,)`` array which is filled with the output values
        for the interpolation.
    """
    cdef unsigned long i=0, n = x.shape[0]
    for i in range(n):
        result_buffer[i] = ys[i,0] + ((x[i]-xs[i,0])*(ys[i,1]-ys[i,0])/(xs[i,1]-xs[i,0]))
    return

cdef void _linterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] result_buffer):
    """
    Perform 1D linear interpolation.

    This function takes an array of `inputs` and interpolates corresponding values
    from `y` based on their positions in the sorted reference array `x`. The function
    assumes that `x` is strictly increasing.

    Parameters
    ----------
    inputs : DTYPE_t[:]
        The query points where interpolation should be performed. If points are outside
        of the abscissa, then extrapolation is used to obtain the values at the points.

    abscissa : DTYPE_t[:]
        The sorted reference x-values (abscissa) at which `y` values are defined.

    ordinate : DTYPE_t[:]
        The corresponding y-values (ordinate) for each x-value. The interpolation
        is based on these values.

    result_buffer : DTYPE_t[:]
        Output buffer where interpolated values are stored.

    Notes
    -----
    - The function uses a custom `searchsorted` implementation to efficiently
      locate the indices where `inputs` should be inserted into `x`.
    - Clipping ensures that interpolation stays within valid bounds.
    - Temporary memory is allocated and freed manually for efficiency.
    - The function assumes `x` is sorted in ascending order.
    """
    # Construct the stencil information and the number of interpolations to perform.
    cdef unsigned long stenc_shape = abscissa.shape[0]
    cdef unsigned long ninterp = inputs.shape[0]
    cdef unsigned long i, iidx

    # Allocate buffers to hold the stencil points around each sample.
    # This requires that we know the two abscissa points bounding the sample and
    # that we know the two ordinate points bounding the sample.
    #
    # Declare the abscissa stencil and the ordinate stencil to hold the
    # values on either side of the evaluation point.
    cdef double* _abscissa_stenc = <double*>malloc(ninterp * 2 * sizeof(double))
    cdef double* _ordinate_stenc = <double*>malloc(ninterp * 2 * sizeof(double))
    cdef double[:, ::1] abscissa_stenc = <double[:ninterp, :2]>_abscissa_stenc
    cdef double[:, ::1] ordinate_stenc = <double[:ninterp, :2]>_ordinate_stenc

    # For each interpolation point, we identify the stencil points and get setup to
    # run the linear interpolation all in one pass.
    for i in range(ninterp):
        # Search for the i-th element of inputs.
        iidx = searchsorted(abscissa, stenc_shape, inputs[i])

        # Clip to the abscissa boundaries.
        if iidx < 1:
            iidx = 1
        elif iidx > stenc_shape - 1:
            iidx = stenc_shape - 1

        # For each of the points we are interpolating, we need to assign the
        # correct stencil.
        abscissa_stenc[i, 0] = abscissa[iidx - 1]
        abscissa_stenc[i, 1] = abscissa[iidx]
        ordinate_stenc[i, 0] = ordinate[iidx - 1]
        ordinate_stenc[i, 1] = ordinate[iidx]

    # Pass to the interpolation function call and execute.
    _linxtnd(inputs,abscissa_stenc,ordinate_stenc,result_buffer)

    # Free the temporary memory.
    free(_abscissa_stenc)
    free(_ordinate_stenc)

cdef void eval_clinterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] bounds,
        DTYPE_t[: ] result_buffer):
    """
    1D linear interpolation with boundary checks and clamping.

    This function checks whether each value in `inputs` lies within the
    user-specified bounding range (`bounds`). If a value is out of `[b0, b1]`,
    it raises a `ValueError`. Otherwise, it clamps the value to the domain
    `[x[0], x[-1]]` to avoid extrapolation. Finally, it calls `_linterp1d`
    to perform the actual linear interpolation, placing the results in
    `result_buffer`.

    Parameters
    ----------
    inputs : DTYPE_t[:]
        The 1D array of query points to be interpolated. Each point is
        checked against `bounds`. If in range, it will be clamped to
        `[x[0], x[-1]]`.

    abscissa : DTYPE_t[:]
        The sorted abscissa array for the interpolation (must be strictly
        increasing). The size is `n_x`.

    ordinate : DTYPE_t[:]
        The 1D array of function values (or likelihood values) defined at
        each point in `x`.

    bounds : DTYPE_t[:]
        A 1D array `[b0, b1]` specifying the valid bounding range for `inputs`.
        If any `inputs[i] < b0` or `inputs[i] > b1`, a `ValueError` is raised.

    result_buffer : DTYPE_t[:]
        A 1D array (size `n_i`) where the interpolation results are stored.
        Must be the same length as `inputs`.

    Raises
    ------
    ValueError
        If any entry in `inputs` lies outside `[b0, b1]`.

    Notes
    -----
    - Values within `[b0, b1]` but outside the domain of `x` are clamped to
      `[x[0], x[-1]]` before interpolation (no extrapolation).
    - Internally calls `_linterp1d` to compute the linear interpolation.
    """
    # Declare static variables.
    cdef unsigned long n_x = abscissa.shape[0], n_i = inputs.shape[0]
    cdef DTYPE_t b0=bounds[0],b1=bounds[1]
    cdef DTYPE_t xb0 = abscissa[0], xb1 = abscissa[n_x-1]
    cdef unsigned long i

    # Enforce the boundary conditions
    for i in range(n_i):
        # Check if the value is out of bounds. If it is, we raise an
        # error instead of trying to coerce anything.
        if (inputs[i]<b0)|(inputs[i]>b1):
            raise ValueError("eval_clinterp1d: inputs out of bounds.")

        # Clip the inputs so that they are not
        # extrapolated during the interpolation.
        inputs[i] = max(xb0,min(inputs[i],xb1))

    # Perform the interpolation.
    _linterp1d(inputs,abscissa,ordinate,result_buffer)

cdef void eval_elinterp1d(
        DTYPE_t[: ] inputs,
        DTYPE_t[: ] abscissa,
        DTYPE_t[: ] ordinate,
        DTYPE_t[: ] bounds,
        DTYPE_t[: ] result_buffer):
    """
    1D linear interpolation with boundary checks (no clamping).

    This function checks whether each value in `inputs` lies within the
    user-specified bounding range (`bounds`). If a value is out of `[b0, b1]`,
    it raises a `ValueError`. It does **not** clamp values to `[x[0], x[-1]]`,
    so queries within `[b0, b1]` but outside the domain of `x` may lead
    to extrapolation when calling `_linterp1d`.

    Parameters
    ----------
    inputs : DTYPE_t[:]
        The 1D array of query points to be interpolated. Each point is
        checked against `bounds`. If in range, it is passed directly to
        `_linterp1d`.

    abscissa : DTYPE_t[:]
        The sorted abscissa array for the interpolation (must be strictly
        increasing). The size is `n_x`.

    ordinate : DTYPE_t[:]
        The 1D array of function values (or likelihood values) defined at
        each point in `x`.

    bounds : DTYPE_t[:]
        A 1D array `[b0, b1]` specifying the valid bounding range for `inputs`.
        If any `inputs[i] < b0` or `inputs[i] > b1`, a `ValueError` is raised.

    result_buffer : DTYPE_t[:]
        A 1D array (size `n_i`) where the interpolation results are stored.
        Must be the same length as `inputs`.

    Raises
    ------
    ValueError
        If any entry in `inputs` lies outside `[b0, b1]`.

    Notes
    -----
    - No clamping is performed, so `_linterp1d` may extrapolate if a point is
      within `[b0, b1]` but outside `[x[0], x[-1]]`.
    - Internally calls `_linterp1d` to compute the linear interpolation.
    """
    # Declare static variables.
    cdef unsigned long n_x = abscissa.shape[0], n_i = inputs.shape[0]
    cdef DTYPE_t b0=bounds[0],b1=bounds[1]
    cdef unsigned long i

    # Enforce the boundary conditions
    for i in range(n_i):
        # Check if the value is out of bounds. If it is, we raise an
        # error instead of trying to coerce anything.
        if (inputs[i]<b0)|(inputs[i]>b1):
            raise ValueError("eval_clinterp1d: inputs out of bounds.")

    # Perform the interpolation.
    _linterp1d(inputs,abscissa,ordinate,result_buffer)

# ============================================================ #
# 2-D Linear interpolators                                     #
# ============================================================ #
cdef void _linterp2d(
        DTYPE_t[: , ::1] inputs,
        DTYPE_t[: , :, ::1] abscissa,
        DTYPE_t[: , ::1] ordinate,
        DTYPE_t[: ] result_buffer):
    """
    Perform 2D bilinear interpolation on a rectilinear grid.

    This function takes an array of 2D query points (`inputs`) and interpolates
    corresponding values from `ordinate` based on their positions in the
    2D sorted reference grid `abscissa`. We assume `abscissa` is of shape
    (Nx, Ny, 2), storing (x, y) coordinates of each grid point, and
    `ordinate` is (Nx, Ny) storing the function values at those grid points.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        2D query points of shape (n_points, 2). Each row is (x_query, y_query).

    abscissa : DTYPE_t[:, :, ::1]
        3D array of shape (Nx, Ny, 2). For each grid cell [i, j],
        `abscissa[i, j, 0]` = x-coordinate, `abscissa[i, j, 1]` = y-coordinate.
        Assumed sorted in ascending order along each axis.

    ordinate : DTYPE_t[:, ::1]
        2D array of shape (Nx, Ny). The function values at each grid point
        `(abscissa[i, j, 0], abscissa[i, j, 1])`.

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]` where interpolated values are stored.

    Notes
    -----
    - Uses two successive 1D linear interpolations (`_linxtnd`) to perform
      bilinear interpolation:
        1) Interpolate `ordinate` in the x-direction (two lines).
        2) Interpolate the results in the y-direction.
    - Assumes strictly ascending `abscissa` in both dimensions so that
      `searchsorted` can identify x- and y-indices.
    - No boundary checks are done here; points are assumed to be clipped
      or validated by caller if out-of-bounds interpolation is undesired.
    """
    # Construct the stencil information and the number of interpolations to perform.
    cdef unsigned long[2] stenc_shape = [abscissa.shape[0],abscissa.shape[1]]
    cdef unsigned long ninterp = inputs.shape[0]
    cdef unsigned long i, iidx, iidy

    # Allocate buffers to hold the stencil points around each sample.
    # This requires that we know the two abscissa points bounding the sample and
    # that we know the two ordinate points bounding the sample.
    #
    # Declare the abscissa stencil and the ordinate stencil to hold the
    # values on either side of the evaluation point.
    #
    # The ordinates are as follows:
    # 0:4 -> LL, LU, UL, UU corners of the domain.
    # 4:6 -> LM = M(LL,LU), UM = (UL,UU)
    cdef double* _abscissa_stenc = <double*>malloc(ninterp * 4 * sizeof(double))
    cdef double* _ordinate_stenc = <double*>malloc(ninterp * 6 * sizeof(double))
    cdef double[:, ::1] abscissa_stenc = <double[:ninterp, :4]>_abscissa_stenc
    cdef double[:, ::1] ordinate_stenc = <double[:ninterp, :6]>_ordinate_stenc

    # For each interpolation point, we identify the stencil points and get setup to
    # run the linear interpolation all in one pass.
    for i in range(ninterp):
        # Search for the i-th element of inputs.
        iidx = searchsorted(abscissa[:,0,0], stenc_shape[0], inputs[i,0])
        iidy = searchsorted(abscissa[0,:,1], stenc_shape[1], inputs[i, 1])

        # Clip to the abscissa boundaries.
        if iidx < 1:
            iidx = 1
        elif iidx > stenc_shape[0] - 1:
            iidx = stenc_shape[0] - 1

        if iidy < 1:
            iidy = 1
        elif iidy > stenc_shape[1] - 1:
            iidy = stenc_shape[1] - 1

        # Construct the abscissa points at the corners.
        abscissa_stenc[i, 0] = abscissa[iidx - 1,iidy-1,0] # bottom left x
        abscissa_stenc[i, 1] = abscissa[iidx,    iidy  ,0] # top right x
        abscissa_stenc[i, 2] = abscissa[iidx - 1,iidy-1, 1] # bottom left y
        abscissa_stenc[i, 3] = abscissa[iidx    ,iidy  , 1] # top right y

        # Construct the ordinate points at the corners.
        # We always go LOW -> HIGH in order of axes.
        # LL, LU, UL, UU
        ordinate_stenc[i, 0] =  ordinate[iidx-1, iidy-1]
        ordinate_stenc[i, 1] =  ordinate[iidx-1, iidy]
        ordinate_stenc[i, 2] =  ordinate[iidx  , iidy-1]
        ordinate_stenc[i, 3] =  ordinate[iidx  , iidy]

    # Compute the midpoint interpolation values along each of the
    # x axes of the grid. This then sets us up to finish with the
    # true results.
    _linxtnd(inputs[:,1],abscissa_stenc[:,2:4],ordinate_stenc[:,:2],ordinate_stenc[:,4])
    _linxtnd(inputs[:,1], abscissa_stenc[:,2:4], ordinate_stenc[:, 2:4], ordinate_stenc[:, 5])
    _linxtnd(inputs[:,0],abscissa_stenc[:,:2],ordinate_stenc[:,4:6],result_buffer)

    # Free the temporary memory.
    free(_abscissa_stenc)
    free(_ordinate_stenc)

cdef void eval_clinterp2d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] ordinate,
        DTYPE_t[:] bounds,           # [bx0, bx1, by0, by1]
        DTYPE_t[:] result_buffer):
    """
    Clamped 2D linear interpolation with boundary checks.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        (n_points, 2) array of (x, y) query points.

    abscissa : DTYPE_t[:, :, ::1]
        (Nx, Ny, 2) array storing the grid of (x, y) coordinates.

    ordinate : DTYPE_t[:, ::1]
        (Nx, Ny) array of values at each grid point in `abscissa`.

    bounds : DTYPE_t[:]
        A 1D array of length 4: [bx0, bx1, by0, by1]. If any point lies
        outside these bounds, an exception is raised. Then the point is
        clamped to the grid domain so `_linterp2d` won’t extrapolate.

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]`, where results are stored.

    Raises
    ------
    ValueError
        If any (x, y) in `inputs` is outside the `[bx0, bx1, by0, by1]`.

    Notes
    -----
    - This function checks the query points against a user-provided bounding
      box. If out of bounds, raises `ValueError`.
    - If inside that bounding box, we then clamp (x, y) to the actual grid
      extremes in `abscissa` to avoid undesired extrapolation.
    - Calls `_linterp2d` to do the actual bilinear interpolation.
    """

    cdef unsigned long i, n_i = inputs.shape[0]
    cdef Py_ssize_t[2] Ngrid = [abscissa.shape[0],abscissa.shape[1]]
    cdef DTYPE_t bx0 = bounds[0]
    cdef DTYPE_t bx1 = bounds[1]
    cdef DTYPE_t by0 = bounds[2]
    cdef DTYPE_t by1 = bounds[3]

    # Grid extremes (lowest corner -> highest corner),
    # assuming ascending in both dimensions:
    cdef DTYPE_t grid_x_min = abscissa[0,    0,    0]
    cdef DTYPE_t grid_x_max = abscissa[Ngrid[0]-1,   Ngrid[1]-1,   0]
    cdef DTYPE_t grid_y_min = abscissa[0,    0,    1]
    cdef DTYPE_t grid_y_max = abscissa[Ngrid[0]-1,  Ngrid[1]-1,   1]

    for i in range(n_i):
        # Check bounding box:
        if (inputs[i, 0] < bx0 or inputs[i, 0] > bx1 or
            inputs[i, 1] < by0 or inputs[i, 1] > by1):
            raise ValueError("eval_clinterp2d: inputs out of bounds.")

        # Clamp to grid domain so we never extrapolate
        inputs[i, 0] = max(grid_x_min, min(inputs[i, 0], grid_x_max))
        inputs[i, 1] = max(grid_y_min, min(inputs[i, 1], grid_y_max))

    _linterp2d(inputs, abscissa, ordinate, result_buffer)


cdef void eval_elinterp2d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, ::1] abscissa,
        DTYPE_t[:, ::1] ordinate,
        DTYPE_t[:] bounds,       # [bx0, bx1, by0, by1]
        DTYPE_t[:] result_buffer):
    """
    2D linear interpolation with boundary checks and no clamping.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        (n_points, 2) array of (x, y) query points.

    abscissa : DTYPE_t[:, :, ::1]
        (Nx, Ny, 2) array storing the grid of (x, y) coordinates.

    ordinate : DTYPE_t[:, ::1]
        (Nx, Ny) array of values at each grid point in `abscissa`.

    bounds : DTYPE_t[:]
        A 1D array of length 4: [bx0, bx1, by0, by1]. If any point lies
        outside these bounds, an exception is raised.

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]`, where results are stored.

    Raises
    ------
    ValueError
        If any (x, y) in `inputs` is outside `[bx0, bx1, by0, by1]`.

    Notes
    -----
    - This function checks the query points against a user-provided bounding
      box. If out of bounds, raises `ValueError`.
    - No clamping is performed. If a point is inside the bounding box but still
      outside the actual grid domain, `_linterp2d` may effectively extrapolate.
    - Calls `_linterp2d` to do the actual bilinear interpolation.
    """
    cdef unsigned long i, n_i = inputs.shape[0]
    cdef DTYPE_t bx0 = bounds[0]
    cdef DTYPE_t bx1 = bounds[1]
    cdef DTYPE_t by0 = bounds[2]
    cdef DTYPE_t by1 = bounds[3]

    # Just check if each query is within [bx0, bx1, by0, by1].
    for i in range(n_i):
        if (inputs[i, 0] < bx0 or inputs[i, 0] > bx1 or
            inputs[i, 1] < by0 or inputs[i, 1] > by1):
            raise ValueError("eval_elinterp2d: inputs out of bounds.")

    _linterp2d(inputs, abscissa, ordinate, result_buffer)

# ============================================================ #
# 3-D Linear interpolators                                     #
# ============================================================ #
cdef void _linterp3d(
        DTYPE_t[: , ::1] inputs,
        DTYPE_t[: , :, :, ::1] abscissa,
        DTYPE_t[: ,:, ::1] ordinate,
        DTYPE_t[: ] result_buffer):
    """
    Perform 3D bilinear interpolation on a rectilinear grid.

    This function takes an array of 3D query points (`inputs`) and interpolates
    corresponding values from `ordinate` based on their positions in the
    3D sorted reference grid `abscissa`. We assume `abscissa` is of shape
    (Nx, Ny, Nz, 2), storing (x, y, z) coordinates of each grid point, and
    `ordinate` is (Nx, Ny, Nz) storing the function values at those grid points.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        2D query points of shape (n_points, 3). Each row is (x_query, y_query, z_query).

    abscissa : DTYPE_t[:, :, ::1]
        4D array of shape (Nx, Ny, Nz, 3). For each grid cell [i, j, k],
        `abscissa[i, j,k , 0]` = x-coordinate, `abscissa[i, j, k, 1]` = y-coordinate.
        Assumed sorted in ascending order along each axis.

    ordinate : DTYPE_t[:, ::1]
        3D array of shape (Nx, Ny, Nz). The function values at each grid point

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]` where interpolated values are stored.

    """
    # Construct the stencil information and the number of interpolations to perform.
    cdef unsigned long[3] stenc_shape = [abscissa.shape[0],abscissa.shape[1], abscissa.shape[2]]
    cdef unsigned long ninterp = inputs.shape[0]
    cdef unsigned long i, iidx, iidy, iidz

    # Allocate buffers to hold the stencil points around each sample.
    # This requires that we know the two abscissa points bounding the sample and
    # that we know the two ordinate points bounding the sample.
    #
    # Declare the abscissa stencil and the ordinate stencil to hold the
    # values on either side of the evaluation point.
    #
    # The ordinate stencil is:
    # Index 0-7: LLL,LLU,LUL,LUU,ULL,ULU,UUL,UUU the actual stencil values
    # Index 8-11: LLM = M(LLL,LLU), LUM = M(LUL,LUU), ULM = M(ULL,ULU), UUM = M(UUL,UUU) the interpolated points
    #   between the x-points.
    # Index 12-13: LMM = M(LLM,LUM), UMM = M(ULM,UUM) the interpolated points between the y-points.
    cdef double* _abscissa_stenc = <double*>malloc(ninterp * 6 * sizeof(double))
    cdef double* _ordinate_stenc = <double*>malloc(ninterp * 14 * sizeof(double))
    cdef double[:, ::1] abscissa_stenc = <double[:ninterp, :6]>_abscissa_stenc
    cdef double[:, ::1] ordinate_stenc = <double[:ninterp, :14]>_ordinate_stenc

    # For each interpolation point, we identify the stencil points and get setup to
    # run the linear interpolation all in one pass.
    for i in range(ninterp):
        # Search for the i-th element of inputs.
        iidx = searchsorted(abscissa[:,0,0,0], stenc_shape[0], inputs[i,0])
        iidy = searchsorted(abscissa[0,:,0,1], stenc_shape[1], inputs[i,1])
        iidz = searchsorted(abscissa[0,0,:,2], stenc_shape[2], inputs[i,2])

        # Clip to the abscissa boundaries.
        if iidx < 1:
            iidx = 1
        elif iidx > stenc_shape[0] - 1:
            iidx = stenc_shape[0] - 1

        if iidy < 1:
            iidy = 1
        elif iidy > stenc_shape[1] - 1:
            iidy = stenc_shape[1] - 1

        if iidz < 1:
            iidz = 1
        elif iidz > stenc_shape[2] - 1:
            iidz = stenc_shape[2] - 1

        # Construct the abscissa points at the corners.
        # We have X0,X1,Y0,Y1,Z0,Z1.
        abscissa_stenc[i, 0] = abscissa[iidx - 1,iidy-1,iidz-1,0] # bottom left x
        abscissa_stenc[i, 1] = abscissa[iidx,    iidy  ,iidz  ,0] # top right x
        abscissa_stenc[i, 2] = abscissa[iidx - 1,iidy-1,iidz-1,1] # bottom left y
        abscissa_stenc[i, 3] = abscissa[iidx,    iidy  ,iidz  ,1] # top right y
        abscissa_stenc[i, 4] = abscissa[iidx - 1,iidy-1,iidz-1,2] # bottom left z
        abscissa_stenc[i, 5] = abscissa[iidx,    iidy  ,iidz  ,2] # top right z

        # Compute the first set of ordinates at the corners.
        # Order is LLL,LLU, ...,
        ordinate_stenc[i, 0] =  ordinate[iidx - 1, iidy - 1, iidz - 1]
        ordinate_stenc[i, 1] =  ordinate[iidx - 1, iidy - 1, iidz]
        ordinate_stenc[i, 2] =  ordinate[iidx - 1, iidy    , iidz - 1]
        ordinate_stenc[i, 3] =  ordinate[iidx - 1, iidy    , iidz]
        ordinate_stenc[i, 4] =  ordinate[iidx    , iidy - 1, iidz - 1]
        ordinate_stenc[i, 5] =  ordinate[iidx    , iidy - 1, iidz]
        ordinate_stenc[i, 6] =  ordinate[iidx    , iidy    , iidz - 1]
        ordinate_stenc[i, 7] =  ordinate[iidx    , iidy    , iidz]


    # Perform the first (z) axis of interpolations (x4)
    _linxtnd(inputs[:, 2],abscissa_stenc[:,  4:], ordinate_stenc[:, :2] , ordinate_stenc[:, 8])
    _linxtnd(inputs[:, 2], abscissa_stenc[:, 4:], ordinate_stenc[:, 2:4], ordinate_stenc[:, 9])
    _linxtnd(inputs[:, 2], abscissa_stenc[:, 4:], ordinate_stenc[:, 4:6], ordinate_stenc[:, 10])
    _linxtnd(inputs[:, 2], abscissa_stenc[:, 4:], ordinate_stenc[:, 6:8], ordinate_stenc[:, 11])

    # Interpolate along the y axis.
    _linxtnd(inputs[:, 1], abscissa_stenc[:, 2:4], ordinate_stenc[:, 8:10], ordinate_stenc[:, 12])
    _linxtnd(inputs[:, 1], abscissa_stenc[:, 2:4], ordinate_stenc[:, 10:12] , ordinate_stenc[:, 13])

    # Interpolate along the x axis.
    _linxtnd(inputs[:, 0], abscissa_stenc[:, :2], ordinate_stenc[:, 12:], result_buffer)

    # Free the temporary memory.
    free(_abscissa_stenc)
    free(_ordinate_stenc)

cdef void eval_clinterp3d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:, :, ::1] ordinate,
        DTYPE_t[:] bounds,           # [bx0, bx1, by0, by1, bz0, bz1]
        DTYPE_t[:] result_buffer):
    """
    Clamped 3D linear interpolation with boundary checks.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        (n_points, 3) array of (x, y, z) query points.

    abscissa : DTYPE_t[:, :, :, ::1]
        (Nx, Ny, Nz, 3) array storing the grid of (x, y, z) coordinates,
        strictly increasing along each axis. Specifically:
            - abscissa[i, j, k, 0] = x-coordinate
            - abscissa[i, j, k, 1] = y-coordinate
            - abscissa[i, j, k, 2] = z-coordinate

    ordinate : DTYPE_t[:, :, ::1]
        (Nx, Ny, Nz) array of values at each grid point in `abscissa`.

    bounds : DTYPE_t[:]
        A 1D array of length 6: [bx0, bx1, by0, by1, bz0, bz1].
        If any point lies outside these bounds, an exception is raised;
        if inside, the point is then clamped to the actual grid domain to
        prevent extrapolation.

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]`, where results are stored
        (one interpolated value for each query).

    Raises
    ------
    ValueError
        If (x, y, z) in `inputs` is outside the bounding box `[bx0, bx1, by0, by1, bz0, bz1]`.

    Notes
    -----
    - This function checks the query points against a user-provided bounding box.
    - If out of bounds, it raises `ValueError`.
    - If in-bounds, each coordinate is clamped to the extremes of the `abscissa` domain
      to avoid extrapolation.
    - Finally, `_linterp3d` is called to do the actual trilinear interpolation.
    """
    cdef unsigned long i, n_i = inputs.shape[0]
    cdef Py_ssize_t[3] Ngrid = [abscissa.shape[0],abscissa.shape[1], abscissa.shape[2]]
    cdef DTYPE_t bx0 = bounds[0]
    cdef DTYPE_t bx1 = bounds[1]
    cdef DTYPE_t by0 = bounds[2]
    cdef DTYPE_t by1 = bounds[3]
    cdef DTYPE_t bz0 = bounds[4]
    cdef DTYPE_t bz1 = bounds[5]

    # Grid extremes (lowest corner -> highest corner),
    # assuming ascending in both dimensions:
    cdef DTYPE_t grid_x_min = abscissa[0,    0,0,   0]
    cdef DTYPE_t grid_x_max = abscissa[Ngrid[0]-1,   Ngrid[1]-1,Ngrid[1]-1,   0]
    cdef DTYPE_t grid_y_min = abscissa[0,    0,0,    1]
    cdef DTYPE_t grid_y_max = abscissa[Ngrid[0]-1,   Ngrid[1]-1,Ngrid[1]-1,   1]
    cdef DTYPE_t grid_z_min = abscissa[0,    0,0,    2]
    cdef DTYPE_t grid_z_max = abscissa[Ngrid[0]-1,   Ngrid[1]-1,Ngrid[1]-1,   2]

    for i in range(n_i):
        # Check bounding box:
        if (inputs[i,0] < bounds[0] or inputs[i,0] > bounds[1] or
            inputs[i,1] < bounds[2] or inputs[i,1] > bounds[3] or
            inputs[i,2] < bounds[4] or inputs[i,2] > bounds[5]):
            raise ValueError("eval_clinterp3d: inputs out of bounds.")

        # Clamp to grid domain so we never extrapolate
        inputs[i, 0] = max(grid_x_min, min(inputs[i,0], grid_x_max))
        inputs[i, 1] = max(grid_y_min, min(inputs[i,1], grid_y_max))
        inputs[i, 2] = max(grid_z_min, min(inputs[i,2], grid_z_max))

    _linterp3d(inputs, abscissa, ordinate, result_buffer)


cdef void eval_elinterp3d(
        DTYPE_t[:, ::1] inputs,
        DTYPE_t[:, :, :, ::1] abscissa,
        DTYPE_t[:, :, ::1] ordinate,
        DTYPE_t[:] bounds,           # [bx0, bx1, by0, by1, bz0, bz1]
        DTYPE_t[:] result_buffer):
    """
    Extrapolated 3D linear interpolation with boundary checks.

    Parameters
    ----------
    inputs : DTYPE_t[:, ::1]
        (n_points, 3) array of (x, y, z) query points.

    abscissa : DTYPE_t[:, :, :, ::1]
        (Nx, Ny, Nz, 3) array storing the grid of (x, y, z) coordinates,
        strictly increasing along each axis. Specifically:
            - abscissa[i, j, k, 0] = x-coordinate
            - abscissa[i, j, k, 1] = y-coordinate
            - abscissa[i, j, k, 2] = z-coordinate

    ordinate : DTYPE_t[:, :, ::1]
        (Nx, Ny, Nz) array of values at each grid point in `abscissa`.

    bounds : DTYPE_t[:]
        A 1D array of length 6: [bx0, bx1, by0, by1, bz0, bz1].
        If any point lies outside these bounds, an exception is raised;
        if inside, the point is then clamped to the actual grid domain to
        prevent extrapolation.

    result_buffer : DTYPE_t[:]
        1D array of length `inputs.shape[0]`, where results are stored
        (one interpolated value for each query).

    Raises
    ------
    ValueError
        If (x, y, z) in `inputs` is outside the bounding box `[bx0, bx1, by0, by1, bz0, bz1]`.

    Notes
    -----
    - This function checks the query points against a user-provided bounding box.
    - If out of bounds, it raises `ValueError`.
    - If in-bounds, each coordinate is clamped to the extremes of the `abscissa` domain
      to avoid extrapolation.
    - Finally, `_linterp3d` is called to do the actual trilinear interpolation.
    """
    cdef unsigned long i, n_i = inputs.shape[0]
    cdef Py_ssize_t[3] Ngrid = [abscissa.shape[0],abscissa.shape[1], abscissa.shape[2]]
    cdef DTYPE_t bx0 = bounds[0]
    cdef DTYPE_t bx1 = bounds[1]
    cdef DTYPE_t by0 = bounds[2]
    cdef DTYPE_t by1 = bounds[3]
    cdef DTYPE_t bz0 = bounds[4]
    cdef DTYPE_t bz1 = bounds[5]

    for i in range(n_i):
        # Check bounding box:
        if (inputs[i,0] < bounds[0] or inputs[i,0] > bounds[1] or
            inputs[i,1] < bounds[2] or inputs[i,1] > bounds[3] or
            inputs[i,2] < bounds[4] or inputs[i,2] > bounds[5]):
            raise ValueError("eval_elinterp3d: inputs out of bounds.")

    _linterp3d(inputs, abscissa, ordinate, result_buffer)
