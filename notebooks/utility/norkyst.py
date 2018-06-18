
# coding: utf-8

# In[9]:


import xarray as xa
import numpy as np
from numba import njit
from numba import jitclass
import numba

# In[10]:


import matplotlib as plt
from numba import types
from numba.extending import typeof_impl
# In[11]:


datafile = 'utility/aggregate_be-4km.nc'
data = xa.open_dataset(datafile, decode_times=False)
data = data.fillna(0)
#data.v.fillna(0)
# In[12]:
#def setNxNY(Nx, Ny):
#    global N
#    N = np.array([Nx,Ny])


class norkystHandler(types.Type):
    def __init__(self, back):
        super(norkystHandler, self).__init__(name='Norkyst')
        limx = -25
        offset = 25
        limy = limx
        time_lim = 72
        time_offset = 72
        self.x = data.X[offset+50:limx] * 1000
        self.y = data.Y[offset:limy] * 1000
        
        if back == True:
            
            self.u = data.u[time_offset - time_lim:time_offset,0,offset:limy,offset+50:limx]
            self.u = self.u[::-1,:,:]
            self.v = data.v[time_offset - time_lim:time_offset,0,offset:limy,offset+50:limx]
            self.v = self.v[::-1,:,:]
            print('t0: ', time_offset - time_lim, 't1: ', time_offset)
            print('size of u:', self.u.shape)
        else:    
            self.u = data.u[time_offset:time_offset + time_lim,0,offset:limy,offset+50:limx]
            self.v = data.v[time_offset:time_offset + time_lim,0,offset:limy,offset+50:limx]
    
        self.dims = self.u.shape
        offsets = np.empty(2)
        offsets[0] = self.x.values[0]
        offsets[1] = self.y.values[0]
        self.offsets = offsets
        self.time_res = data.time.values[1] - data.time.values[0]
        self.x_res = self.x.values[1] - self.x.values[0]
        self.y_res = self.y.values[1] - self.y.values[0]
   
    def __call__(self, Nx, Ny):
        N_ = np.empty(2)
        N_[0] = Nx
        N_[1] = Ny
        self.N = N_



    def grid_of_particles(self, Nx, Ny, wx=2, wy = 1.):
        
        
        #Nx = 1*int(np.sqrt(N))
        #Ny = 1*int(np.sqrt(N))
        x  = np.linspace(self.x.values[0] + 15*self.x_res, self.x.values[-1] - 15*self.x_res, Nx)
        y  = np.linspace(self.y.values[0] + 15*self.y_res, self.y.values[-1] - 15*self.y_res, Ny)
        y, x = np.meshgrid(y, x)
        return np.array([x, y])
    

    #@njit(['float64[:,:](Norkyst, float64[:,:], float64)'], parallel=True)
    def f(self, X, t):
        ret = f_in3(X, t, self.offsets, self.x_res, self.y_res, self.time_res, self.u, self.v, self.dims, xcord = self.x, ycord = self.y)
        return ret

from scipy.interpolate import RectBivariateSpline as RBS    
import numba  

#@numba.jit(nopython=False)
def interp_wrap(xc,yc,t_0,X_,u):
    orde = 3

    interp_ = RBS(yc.values, xc.values, u.values[t_0,:,:], kx=orde, ky=orde)
    return interp_.ev(X_[1,:], X_[0,:])

#@njit()
def f_in3(X, t, offsets, x_res, y_res, time_res, u, v, dims, xcord=None, ycord=None):

    t_0 = np.abs(int(t/time_res))

    Shape = X.shape
    dims = len(Shape)
    temp = X.reshape((2, -1))
    X = temp
    
    ret1 = (interp_wrap(xcord, ycord,t_0, X, u))

    ret1 = np.nan_to_num(ret1)
    
    ret2 = (interp_wrap(xcord, ycord,t_0, X, v))
    ret2 = np.nan_to_num(ret2)
    if len(Shape) > 2:
        X = X.reshape(Shape)
        ret1 = ret1.reshape(Shape[1], Shape[2])
        ret2 = ret2.reshape(Shape[1], Shape[2])
    return np.array([ret1, ret2])
# @njit(parallel=True)
# def doublegyre(X, t, A, e, w):
#     a = e * np.sin(w*t)
#     b = 1 - 2.*e*np.sin(w*t)
#     f = a*X[0,:]**2 + b*X[0,:]
#     v = np.empty(X.shape)
#     v[0,:] = -np.pi*A*np.sin(np.pi*f) * np.cos(np.pi*X[1,:])                    # x component of velocity
#     v[1,:] =  np.pi*A*np.cos(np.pi*f) * np.sin(np.pi*X[1,:]) * (2.*a*X[0,:] + b) # y component of velocity
#     return v


# # Wrapper function to pass to integrator
# # X0 is a two-component vector [x, y]
# @njit
# def f(X, t):
#     # Parameters of the velocity field
#     A = A_       # A
#     e = e_      # epsilon
#     w = 2.*np.pi/10.  # omega
#     return doublegyre(X, t, A, e, w)

