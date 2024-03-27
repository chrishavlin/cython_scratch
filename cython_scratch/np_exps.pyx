cimport numpy as np
cimport cython

cpdef float do_a_thing(float a):
    return a * 2.0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def process_2d_array(np.float64_t[:,:] x):
    cdef int i, j, ni, nj
    cdef np.float64_t sumval
    ni = x.shape[0]
    nj = x.shape[1]

    sumval = 0.0

    with nogil:
        for i in range(ni):
            for j in range(nj):
                sumval += x[i, j]

    return sumval

