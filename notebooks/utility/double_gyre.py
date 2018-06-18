
# coding: utf-8

# In[ ]:


import numpy as np
from numba import njit
# The double gyre velocity field

#A_ = 0.1
#e_ = 0.1

def setGyreVars(A, e):
    global A_
    A_ = A
    global e_
    e_ = e
    return A_, e_


def grid_of_particles(Nx, Ny, wx=2, wy = 1.):
    # Create a grid of N (approximately) evenly spaced particles
    # covering a rectangular patch of width wx and height wy
    # centered on the point (1.0, 0.5)
    #Nx = 2*int(np.sqrt(N/2))
    #Ny = 1*int(np.sqrt(N/2))
    x  = np.linspace(1.0-wx/2, 1.0+wx/2, Nx)
    y  = np.linspace(0.5-wy/2, 0.5+wy/2, Ny)
    y, x = np.meshgrid(y, x)
    return np.array([x, y])

@njit(parallel=True)
def doublegyre(X, t, A, e, w):
    a = e * np.sin(w*t)
    b = 1 - 2.*e*np.sin(w*t)
    f = a*X[0,:]**2 + b*X[0,:]
    v = np.empty(X.shape)
    v[0,:] = -np.pi*A*np.sin(np.pi*f) * np.cos(np.pi*X[1,:])                    # x component of velocity
    v[1,:] =  np.pi*A*np.cos(np.pi*f) * np.sin(np.pi*X[1,:]) * (2.*a*X[0,:] + b) # y component of velocity
    return v


# Wrapper function to pass to integrator
# X0 is a two-component vector [x, y]
@njit
def f(X, t):
    # Parameters of the velocity field
    A =  A_       # A
    e =  e_      # epsilon
    w = 2.*np.pi/10.  # omega
    return doublegyre(X, t, A, e, w)


# 4th order Runge-Kutta integrator
# X0 is a two-component vector [x, y]

