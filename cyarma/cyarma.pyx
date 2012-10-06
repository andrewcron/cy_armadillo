import numpy as np
cimport numpy as np
cimport cython

from cython.operator cimport dereference as deref

from libcpp cimport bool

cdef extern from "Armadillo" namespace "arma":
    # matrix class (double)
    cdef cppclass mat:
        mat(double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict)
        mat()
        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero
        # fuctions
        mat i() #inverse
        mat t() #transpose
        vec diag()
        vec diag(int)
        fill(double)
        void raw_print(char*)
        void raw_print()
        #print(char)
        #management
        mat reshape(int, int)
        mat resize(int, int)
        double * memptr()
        # opperators
        double operator[](int)
        mat operator*(mat)
        mat operator*(double)
        mat operator+(mat)
        mat operator+(double)
        mat operator/(double)
        #etc

    # vector class (double)
    cdef cppclass vec:
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict)
        # attributes
        int n_elem
        # opperators
        double operator[](int)
        # functions
        double * memptr()
    ## TODO: cude class (double)
        
    # Armadillo Linear Algebra tools
    cdef bool chol(mat R, mat X) # preallocated result
    cdef mat chol(mat X) # new result
    cdef bool inv(mat R, mat X)
    cdef mat inv(mat X)
    cdef bool solve(vec x, mat A, vec b)
    cdef vec solve(mat A, vec b)
    cdef bool solve(mat X, mat A, mat B)
    cdef mat solve(mat A, mat B)
    cdef bool eig_sym(vec eigval, mat eigvec, mat B)
    cdef bool svd(mat U, vec s, mat V, mat X, method = "standard")
    cdef bool lu(mat L, mat U, mat P, mat X)
    cdef bool lu(mat L, mat U, mat X)
    cdef mat pinv(mat A)
    cdef bool pinv(mat B, mat A)
    cdef bool qr(mat Q, mat R, mat X)
    




#from cyarma cimport *

##### Tools to convert numpy arrays to armadillo arrays ######
cdef mat numpy_to_mat(np.ndarray[np.double_t, ndim=2] X):
    cdef mat *aR
    if not X.flags.f_contiguous:
        X = X.copy(order="F")
    aR  = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)

    return deref(aR)

cdef vec numpy_to_vec(np.ndarray[np.double_t, ndim=1] x):
    cdef vec *ar = new vec(<double*> x.data, x.shape[0], False, True)
    return deref(ar)


##### Converting back to python arrays, must pass preallocated memory or None
# all data will be copied since numpy doesn't own the data and can't clean up
# otherwise. Maybe this can be improved. #######
@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=2] mat_to_numpy(mat X, np.ndarray[np.double_t, ndim=2] D):
    cdef double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty((X.n_rows, X.n_cols), dtype=np.double, order="F")
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_rows*X.n_cols):
        Dptr[i] = Xptr[i]
    return D

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=1] vec_to_numpy(mat X, np.ndarray[np.double_t, ndim=1] D):
    cdef double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty(X.n_elem, dtype=np.double)
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_elem):
        Dptr[i] = Xptr[i]
    return D

### A few wrappers for much needed numpy linalg functionality using armadillo
cpdef np_chol(np.ndarray[np.double_t, ndim=2] X):
    # initialize result numpy array
    cdef np.ndarray[np.double_t, ndim=2] R = \
         np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
    # wrap them up in armidillo arrays
    cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
    cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
    chol(deref(aR), deref(aX))
    
    return R

cpdef np_inv(np.ndarray[np.double_t, ndim=2] X):
    # initialize result numpy array
    cdef np.ndarray[np.double_t, ndim=2] R = \
         np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
    # wrap them up in armidillo arrays
    cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
    cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
    inv(deref(aR), deref(aX))
    
    return R

def np_eig_sym(np.ndarray[np.double_t, ndim=2] X):
    # initialize result numpy array
    cdef np.ndarray[np.double_t, ndim=2] R = \
         np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
    cdef np.ndarray[np.double_t, ndim=1] v = \
         np.empty(X.shape[0], dtype=np.double)
    # wrap them up in armidillo arrays
    cdef mat *aX = new mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
    cdef mat *aR  = new mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    cdef vec *av = new vec(<double*> v.data, v.shape[0], False, True)

    eig_sym(deref(av), deref(aR), deref(aX))

    return [v, R]


# def example(np.ndarray[np.double_t, ndim=2] X):
#     cdef mat aX = numpy_to_mat(X)
#     cdef mat XX = aX.t() * aX
#     cdef mat ch = chol(XX)
#     ch.raw_print()
#     cdef np.ndarray[np.double_t,ndim=2] Y = mat_to_numpy(ch, None)
#     return Y
