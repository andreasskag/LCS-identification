# Julius Bier Kirkegaard 2017
# Based on "Computing Discrete Frechet Distance" by "Thomas Eiter and Heikki Mannila".
#cython: boundscheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, cos, sin

cdef double d(double p, double q):
    return (p - q)*(p - q)


cdef double c(int i, int j, double* P, double* Q, double* ca, int N):
    cdef int index = i*N+j
    cdef double d_t = d(P[i], Q[j])
    cdef double m1, m2

    if ca[index] > -1:
        return ca[index]

    elif i == 0 and j == 0:
        ca[index] = d(P[i], Q[j])

    elif i > 0 and j == 0:
        if d_t > ca[(i-1)*N+j]:
            ca[index] = d_t
        else:
            ca[index] = ca[(i-1)*N+j]

    elif i == 0 and j > 0:
        if d_t > ca[i*N+(j-1)]:
            ca[index] = d_t
        else:
            ca[index] = ca[i*N+(j-1)]

    elif i > 0 and j > 0:
        m1 = c(i - 1, j, P, Q, ca, N)
        m2 = c(i, j - 1, P, Q, ca, N)
        if m2 < m1:
            m1 = m2
        m2 = c(i - 1, j - 1, P, Q, ca, N)
        if m2 < m1:
            m1 = m2
        if d_t > m1:
            ca[index] = d_t
        else:
            ca[index] = m1
    else:
        ca[index] = 1e50

    return ca[index]

def frechet(np.ndarray[np.float64_t,ndim=1,
                    negative_indices=False,
                    mode='c'] P,
            np.ndarray[np.float64_t,ndim=1,
                    negative_indices=False,
                    mode='c'] Q):

    cdef np.ndarray[np.float64_t,ndim=2,
                    negative_indices=False,
                    mode='c'] ca = np.zeros((len(P), len(Q))) - 1

    return c(len(P)-1, len(Q)-1, &P[0], &Q[0], &ca[0,0], ca.shape[0])