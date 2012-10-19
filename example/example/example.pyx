import numpy as np
cimport numpy as np

include "cyarma.pyx"

def example(np.ndarray[np.double_t, ndim=2] X):

    cdef mat aX = numpy_to_mat(X)

    cdef mat XX = aX.t() * aX
    cdef mat ch = chol(XX)
    ch.raw_print()
    print np.linalg.cholesky(np.dot(X.T,X))
    cdef np.ndarray[np.double_t,ndim=2] Y = mat_to_numpy(ch, None)

    cdef double ctest[10][10]
    cdef double [:,:] test = ctest
    cdef int i, j
    for i in range(10):
        for j in range(10):
            test[i,j] = i+j
    cdef mat *test2 = new mat(<double*> ctest, 10, 10, False, True)
    test2.raw_print()
    
    return Y
