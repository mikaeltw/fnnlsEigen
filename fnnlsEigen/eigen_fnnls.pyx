# cython: language_level = 3

from numpy cimport ndarray as cndarray
from numpy cimport float64_t, float32_t


cdef extern from "eigen_fnnls_pywrap.h" namespace "fnnls_python_wrapper":
    cdef cndarray call_fnnls_solver[T](cndarray array_Z,
                                       cndarray array_x,
                                       int max_iterations,
                                       T tolerance) except +

    cdef cppclass StorePreCompute[T]:
        cndarray call_fnnls_solver_precompute(cndarray array_Z,
                                              cndarray array_x,
                                              int max_iterations,
                                              T tolerance) except +


cdef class CachePreComputeNNLS:
    """
    This class holds the latest parameter matrix and its symmetric rankUpdated product.
    This enables faster solving since the costly Z^TZ product can be reused. This class
    works with 64 bit floating type.

    """

    cdef StorePreCompute[float64_t]* precompute_obj

    def __cinit__(self):
        self.precompute_obj = new StorePreCompute[float64_t]()
        if self.precompute_obj == NULL:
            raise MemoryError('Not enough memory.')

    def __dealloc__(self):
        del self.precompute_obj

    def fnnls(self,
              cndarray[float64_t, ndim=2] array_Z,
              cndarray[float64_t, ndim=1] array_x,
              int max_iterations=0,
              float64_t tolerance=-1.0):
        """
        This class function solves the same problem as `fnnls`, but the previous
        parameter matrix it was called with and its symmetric rankUpdate ZTZ, is stored
        in its c++ class instance in order to save time not recomputing it should it be
        the same as last time.

        This implementation is optimised for speed when Z is intermediately sparse.

        Parameters
        ----------
        Z : ndarray(M, N) of dtype=np.float64
            Z is the coefficient matrix.
        x : ndarray(M) of dtype=np.float64
            x is the vector of independent parameters.
        max_iterations : int
            max_iterations of the iterative nnls solver. Defaults to 3 * array_Z.shape[1].
        tolerance : np.float64
            Precision per element, defaults to machine epsilon of float times array_Z.shape[1].

        Returns
        -------
        d : ndarray(N) of dtype=np.float64
            d is the solution to min_{d>=0} ||x - Zd||_2.

        """

        return self.precompute_obj.call_fnnls_solver_precompute(array_Z, array_x, max_iterations, tolerance)


# Exposed python fnnls solver
def fnnls(cndarray[float64_t, ndim=2] array_Z,
          cndarray[float64_t, ndim=1] array_x,
          int max_iterations=0,
          float64_t tolerance=-1.0):
    """
    Implementation of the fast non-negative least squares algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong (1997). This function uses floating point 64-bit precision.

    This implementation is optimised for speed when Z is intermediately sparse.

    Parameters
    ----------
    Z : ndarray(M, N) of dtype=np.float64
        Z is the coefficient matrix.
    x : ndarray(M) of dtype=np.float64
        x is the vector of independent parameters.
    max_iterations : int
        max_iterations of the iterative nnls solver. Defaults to 3 * array_Z.shape[1].
    tolerance : np.float64
        Precision per element, defaults to machine epsilon of float times array_Z.shape[1].

    Returns
    -------
    d : ndarray(N) of dtype=np.float64
        d is the solution to min_{d>=0} ||x - Zd||_2.

    """

    return call_fnnls_solver(array_Z, array_x, max_iterations, tolerance)


cdef class CachePreComputeNNLSf:
    """
    This class holds the latest parameter matrix and its symmetric rankUpdated product.
    This enables faster solving since the costly Z^TZ product can be reused. This class
    works with 32 bit floating type.

    """

    cdef StorePreCompute[float32_t]* precompute_obj

    def __cinit__(self):
        self.precompute_obj = new StorePreCompute[float32_t]()
        if self.precompute_obj == NULL:
            raise MemoryError('Not enough memory.')

    def __dealloc__(self):
        del self.precompute_obj

    def fnnls(self,
              cndarray[float32_t, ndim=2] array_Z,
              cndarray[float32_t, ndim=1] array_x,
              int max_iterations=0,
              float32_t tolerance=-1.0):
        """
        This class function solves the same problem as `fnnls`, but the previous
        parameter matrix it was called with and its symmetric rankUpdate ZTZ, is stored
        in its c++ class instance in order to save time not recomputing it should it be
        the same as last time.

        This implementation is optimised for speed when Z is intermediately sparse.

        Parameters
        ----------
        Z : ndarray(M, N) of dtype=np.float32
            Z is the coefficient matrix.
        x : ndarray(M) of dtype=np.float32
            x is the vector of independent parameters.
        max_iterations : int
            max_iterations of the iterative nnls solver. Defaults to 3 * array_Z.shape[1].
        tolerance : float32
            Precision per element, defaults to machine epsilon of float times array_Z.shape[1].

        Returns
        -------
        d : ndarray(N) of dtype=np.float32
            d is the solution to min_{d>=0} ||x - Zd||_2.

        """

        return self.precompute_obj.call_fnnls_solver_precompute(array_Z, array_x, max_iterations, tolerance)


# Exposed python fnnls solver
def fnnlsf(cndarray[float32_t, ndim=2] array_Z,
           cndarray[float32_t, ndim=1] array_x,
           int max_iterations=0,
           float32_t tolerance=-1.0):
    """
    Implementation of the fast non-negative least squares algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong (1997). This function uses floating point 32-bit precision.

    This implementation is optimised for speed when Z is intermediately sparse.

    Parameters
    ----------
    Z : ndarray(M, N) of dtype=np.float32
        Z is the coefficient matrix.
    x : ndarray(M) of dtype=np.float32
        x is the vector of independent parameters.
    max_iterations : int
        max_iterations of the iterative nnls solver. Defaults to 3 * array_Z.shape[1].
    tolerance : float32
        Precision per element, defaults to machine epsilon of float times array_Z.shape[1].

    Returns
    -------
    d : ndarray(N) of dtype=np.float32
        d is the solution to min_{d>=0} ||x - Zd||_2.

    """

    return call_fnnls_solver(array_Z, array_x, max_iterations, tolerance)
