import numpy as np
cimport numpy as np

include "cyarma.pyx"

def example(np.ndarray[np.double_t, ndim=2] X):
    cdef mat aX = numpy_to_mat(X)
    cdef mat XX = aX.t() * aX
    cdef mat ch = chol(XX)
    ch.raw_print()
    cdef np.ndarray[np.double_t,ndim=2] Y = mat_to_numpy(ch, None)
    return Y
