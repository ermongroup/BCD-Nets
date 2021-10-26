#include <algorithm>
#include <cmath>
#include "rectangular_lsap.h"
#include <vector>
#include <stdint.h>

from libc.stdlib cimport malloc, free
import numpy as np
from libc.stdint cimport int64_t
from libc.stdint cimport intptr_t
from cython.parallel import prange
import cython

cdef extern from "rectangular_lsap.cpp":
    int solve_rectangular_linear_sum_assignment(intptr_t nr, double* input_cost, int* col4row) nogil


DTYPE = np.float32


def compute(double[:, :, ::1] cost_batch):
    
    cdef Py_ssize_t n_batch = cost_batch.shape[0]
    cdef Py_ssize_t dim = cost_batch.shape[1]

    result = np.zeros((n_batch, dim), dtype=np.int32)
    cdef int[:, :] result_view = result

    # cdef double[:, ::1] cost_two_d_view = cost_batch

    cdef Py_ssize_t n

    with cython.boundscheck(False):
        with cython.wraparound(False):
            for n in range(n_batch):
                solve_rectangular_linear_sum_assignment(dim, &cost_batch[n, 0, 0], &result_view[n, 0])

    return result


def compute_parallel(double[:, :, ::1] cost_batch):
    
    cdef Py_ssize_t n_batch = cost_batch.shape[0]
    cdef Py_ssize_t dim = cost_batch.shape[1]

    result = np.zeros((n_batch, dim), dtype=np.int32)
    cdef int[:, :] result_view = result

    # cdef double[:, ::1] cost_two_d_view = cost_batch

    cdef Py_ssize_t n

    with cython.boundscheck(False):
        with cython.wraparound(False):
            for n in prange(n_batch, nogil=True):
                solve_rectangular_linear_sum_assignment(dim, &cost_batch[n, 0, 0], &result_view[n, 0])

    return result
