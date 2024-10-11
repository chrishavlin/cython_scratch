from libc.math cimport isfinite

cimport numpy as np
cimport cython
import numpy as np

cdef extern from "numpy/npy_math.h":
    double NPY_PI

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _process_2d_manual(np.float64_t[:,:] x, np.float64_t[:,:] xout):
    cdef int i, j, ni, nj
    ni = x.shape[0]
    nj = x.shape[1]
    for i in range(ni):
        for j in range(nj):
            xout[i,j] = x[i,j] * 2.0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process_2d_array(np.float64_t[:,:] x):
    cdef np.float64_t[:,:] xout
    xout = np.full((x.shape[0], x.shape[1]), np.nan, dtype=np.float64)
    _process_2d_manual(x, xout)
    return xout


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _process_3d_manual(np.float64_t[:,:,:] x, np.float64_t[:,:,:] xout):
    cdef int i, j, ni, nj, k, nk
    ni = x.shape[0]
    nj = x.shape[1]
    nk = x.shape[2]
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                xout[i,j,k] = x[i,j,k] * 2.0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process_3d_array(np.float64_t[:,:,:] x):
    cdef np.float64_t[:,:,:] xout
    xout = np.full((x.shape[0], x.shape[1], x.shape[2]), np.nan, dtype=np.float64)
    _process_3d_manual(x, xout)
    return xout


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _process_1d_memview(np.float64_t[:] x1d, np.float64_t[:] x1dout, int n) noexcept nogil:
    # assumes x1d, x1dout already flattened
    cdef int i
    for i in range(n):
        x1dout[i] = x1d[i] * 2.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process_nd_array(np.ndarray x):
    cdef int nsize, ndim, idim
    cdef np.ndarray[np.float64_t, ndim=1] x1d, xout
    cdef np.int64_t[:] xshape

    x1d = x.reshape(-1)
    nsize = x1d.size

    # need to provide an initialize array to fill
    xout = np.full((x1d.size,), np.nan, dtype=np.float64)
    _process_1d_memview(x1d, xout, nsize)

    ndim = x.ndim
    if ndim == 1:
        return xout

    # need to reshape: just using xout.reshape(x.shape) fails
    # because the tuple unpacking is not supported in
    # cython. so, build an array from the tuple and use that.
    xshape = np.zeros((ndim,), dtype=np.int64)
    for idim in range(ndim):
        xshape[idim] = x.shape[idim]
    return xout.reshape(xshape)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process_nd_array_v2(np.ndarray x):
    cdef int nsize, ndim, idim
    cdef np.ndarray[np.float64_t, ndim=1] x1d, xout1d
    cdef np.int64_t[:] xshape

    x1d = x.reshape(-1)
    nsize = x1d.size

    # create the array to fill as an nd array,
    # pass a reshaped view down
    ndim = x.ndim
    xshape = np.zeros((ndim,), dtype=np.int64)
    for idim in range(ndim):
        xshape[idim] = x.shape[idim]
    xout = np.full(xshape, np.nan, dtype=np.float64)
    xout1d = xout.reshape(-1)

    _process_1d_memview(x1d, xout1d, nsize)

    return xout


def sum_no_finite(ntimes):
    sum_n_times(ntimes)

def sum_check_finite(ntimes):
    sum_n_times_w_finite(ntimes)

cdef void sum_n_times(unsigned long long n) noexcept nogil:
    cdef np.float64_t val
    cdef unsigned long long i

    val = 0.0
    for i in range(n):
        val += 1.0

cdef void sum_n_times_w_finite(unsigned long long n) noexcept nogil:
    cdef np.float64_t val
    cdef unsigned long long i

    val = 0.0
    for i in range(n):
        if isfinite(val):
            val += 1.0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def compare_modulo_with_pi(np.float64_t test_val):

    cdef double NPY_2PI = 2 * NPY_PI
    cdef np.float64_t twoPI = 2 * NPY_PI
    cdef np.float64_t other_2pi = 2 * np.pi
    cdef np.float64_t result_1, result_2, result_3, result_4

    with nogil:
        result_1 = test_val % NPY_2PI

    result_2 = test_val % other_2pi
    result_3 = test_val % NPY_2PI

    with nogil:
        result_4 = test_val % twoPI

    return result_1, result_2, result_3, result_4

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def compare_modulo_with_pi_no_c_div(np.float64_t test_val):

    cdef double NPY_2PI = 2 * NPY_PI
    cdef np.float64_t twoPI = 2 * NPY_PI
    cdef np.float64_t other_2pi = 2 * np.pi
    cdef np.float64_t result_1, result_2, result_3, result_4

    with nogil:
        result_1 = test_val % NPY_2PI

    result_2 = test_val % other_2pi
    result_3 = test_val % NPY_2PI

    with nogil:
        result_4 = test_val % twoPI

    return result_1, result_2, result_3, result_4