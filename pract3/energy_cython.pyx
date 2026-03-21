# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np

def energy_calc_cython(double[:] power, double[:] temp):
    cdef int n = power.shape[0]
    cdef double[:] res = np.zeros(n, dtype=np.float64)
    cdef int i
    cdef double diff

    for i in range(n):
        diff = temp[i] - 20.0
        if diff < 0:
            diff = -diff
        res[i] = power[i] * (1.0 + 0.05 * diff)
        
    return np.asarray(res)
