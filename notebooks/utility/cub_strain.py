# Numpy
import numpy as np



# Numba (JiT)
from numba import jit

# (Primitive) timing functionality
import time 

# Multiprocessing:
import multiprocessing as mp

# Spline interpolation:
from scipy.interpolate import RectBivariateSpline, interp1d

# Check whether folders exist or not, necessary
# for storing advected states:
import os
import errno

# Display progress bars:
from ipywidgets import FloatProgress
from IPython.display import display

from concurrent.futures import ProcessPoolExecutor, as_completed

Nx = 0
Ny = 0
n_ = 0

def setNXNY(Nx_, Ny_, N_):
    global Nx, Ny, n
    Nx = Nx_
    Ny = Ny_
    n_ = N_


@jit(nopython = True)
def thomas(a, b, c, d):
    # Implementation of the Thomas algorithm
    # for solving trilinear equation system,
    # used in calculating spline coefficients
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    # modified because in our case, a, b and c are scalars
    # (due to assumption of constant grid spacing)
    
    # The behaivour of this function should be exactly equivalent to
    # N = len(d)
    # A = np.diag(a*np.ones(N-1), -1) + np.diag(b*np.ones(N), 0) + np.diag(c*np.ones(N-1), 1)
    # x = np.linalg.solve(A, d)
    # return x
    
    N = len(d)
    c_ = np.empty(N)
    d_ = np.empty(N)
    x  = np.empty(N)
    c_[0] = c   /b
    d_[0] = d[0]/b
    for i in range(1, N):
        f = (b - a*c_[i-1])
        c_[i] = c/f
        d_[i] = (d[i] - a*d_[i-1])/f
    x[-1] = d_[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_[i] - c_[i]*x[i+1]
    return x

@jit(nopython = True)
def unispline(xs, fs, x):
    # takes arrays with values for xs and f(xs)
    # returns interpolated value of f(x)
    # Assume constant spacing
    dx = xs[1] - xs[0]
    # Solve to find weights
    fs_ = 3*(fs[2:] - 2*fs[1:-1] + fs[:-2])/dx
    S = np.zeros(fs_.size + 2)
    S[1:-1] = thomas(dx, 4*dx, dx, fs_)
    # Identify index of left node in interval
    i = np.int32((x - xs[0]) / dx)
    # Calculate relative position in interval
    x_ = x - xs[i]
    # Interpolate to position
    a = (S[i+1]-S[i])/(3*dx)
    b = S[i]
    c = (fs[i+1]-fs[i])/dx - dx*(2*S[i] + S[i+1])/3
    d = fs[i]
    return a*x_**3 + b*x_**2 + c*x_ + d

@jit(nopython = True)
def bispline(xs, ys, fs, x, y):
    # takes rank 1 arrays with values for xs, ys
    # and rank 2 array with values for f(xs, ys)
    # returns inerpolated value of f(x, y)
    
    # use unispline repeatedly along dimensions
    # first along x, to obtain rank 2 array with values for f(x, ys)
    fxs = np.empty((ys.size))
    for i in range(ys.size):
        fxs[i] = unispline(xs, fs[:,i], x)
    
    # then, interpolate along y and return f(x, y)
    fxy = unispline(ys, fxs, y)
    return fxy

class CubicSpecial():
    def __init__(self, xc, yc, xi, n_):
        self.xc = xc
        self.yc = yc
        self.dx = xc[1] - xc[0]
        self.dy = yc[1] - yc[0]
        self.xi = xi
        self.Nx = len(xc)
        self.Ny = len(yc)
        self.n = n_
        self.fold = None
        
    def __call__(self, X, t):
        
        N  = self.n
        N1 = int(N/2 - 1)
        N2 = int(N/2 + 1)
        # Calculate indices for lower left corner in cell
        i = np.floor((X[0] - self.xc[0]) / self.dx).astype(np.int32)
        j = np.floor((X[1] - self.yc[0]) / self.dy).astype(np.int32)
        
        # If outside the domain, stop
        if (i >= self.Nx - N2) or (j >= self.Ny - N2) or (i < N1) or (j < N1):
            raise IndexError
        
        # Use the lower left corner as reference, calculate
        # the rotation of the other vectors, and rotate by 180
        # degrees if required (due to orientational discontinuity)
        subxi = self.xi[:,i-N1:i+N2, j-N1:j+N2].copy()
        dotp = np.sum(subxi[:,0,0].reshape(2,1,1) * subxi, axis = 0)
        subxi[:, dotp < 0] =  -subxi[:, dotp < 0]
        
        V = np.zeros(2)
        # Cubic interpolation
        V[0] = bispline(self.xc[i-N1:i+N2], self.yc[j-N1:j+N2], subxi[0,:], X[0], X[1])
        V[1] = bispline(self.xc[i-N1:i+N2], self.yc[j-N1:j+N2], subxi[1,:], X[0], X[1])
                
        # Check orientation against previous vector
        if self.fold is None:
            return V
        else:
            # If dot product is negative, flip direction
            return V * np.sign(np.dot(V, self.fold))


def rk4(X, t, h, f, scale = 1):
    k1 = scale*f(X,          t)
    k2 = scale*f(X + k1*h/2, t + h/2)
    k3 = scale*f(X + k2*h/2, t + h/2)
    k4 = scale*f(X + k3*h,   t + h)
    return X + h*(k1 + 2.*k2 + 2.*k3 + k4) / 6.


def half_strainline(x0, Tmax, h, f, xc, yc, lambdas, AB_interp, grid, pm, max_notAB = 0.3, t=0):
    # Re-initialise the f-function
    f.fold = None

    Nt = np.abs(int((Tmax-t) / h))
    xs = np.zeros((2, Nt))
    xs[:,0] = x0
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    
    # Buffer zone outside domain
    xbuf = 2*(grid[0,1,0] - grid[0,0,0])
    ybuf = 2*(grid[1,0,1] - grid[1,0,0])
    
    n_ = f.n
    bound_xlow = grid[0,int(n_/1),0]
    bound_xhigh = grid[0,-int(n_/1),0]
    bound_ylow = grid[1,0,int(n_/1)]
    bound_yhigh = grid[1,0,-int(n_/1)]
    
    # Parameters of strainline
#     AB_interp = RectBivariateSpline(xc, yc, ABtrue)
    length = 0.0
    notABlength = 0.0
    mulambda = 0.0

    for n in range(1, Nt):
        lamb1 = lambdas[0](xs[0,n-1], xs[1,n-1])
        lamb2 = lambdas[1](xs[0,n-1], xs[1,n-1])
        
        scaling = (lamb1 - lamb2) / (lamb1 + lamb2)
        scaling = scaling[0,0]**2
        
        if scaling > 1: 
            scaling = 1
        elif scaling < 1e-5:
            break #stuff is singular  
        
        f.fold = f(xs[:,n-1], t)
        try:
            xs[:,n] = rk4(xs[:,n-1], t, pm*h, f, scale = 1)
        except IndexError:
            break
        if xs[0,n] < (bound_xlow-xbuf) or (bound_xhigh+xbuf) < xs[0,n] or xs[1,n] < (bound_ylow-ybuf) or (bound_yhigh+ybuf) < xs[1,n]:
            break
        if notABlength > max_notAB:
            break

        # increment length
        dl = np.sqrt(np.sum((xs[:,n] - xs[:,n-1])**2))
        length += dl
        # calculate closest grid point
        ##i = np.floor(((xs[0,n]+dx/2) - xc[0]) / dx).astype(np.int32)
        ##j = np.floor(((xs[1,n]+dy/2) - yc[0]) / dy).astype(np.int32)
        # Use this to look up lambda2, and add to running total
        ##mulambda += lambda2[i,j] * dl
        mulambda += lambdas[1](xs[0,n], xs[1,n]) * dl
        # Check if A and B are satisfied
        if AB_interp(xs[0,n], xs[1,n]) > 0.10:
            notABlength = 0
        else:
            notABlength += dl

    if length > 0:
        mulambda = mulambda / length
    else:
        mulambda = 0.0
    return xs[:,:n], length, mulambda

def strainline(x0, Tmax, h, f, xc, yc, lambdas, AB_interp, grid, max_notAB = 0.3, t=0):
    num_iters = x0.shape[1]
    lines = []
    mulambdas = []
    lengths = []
    for s in range(num_iters):
        line1, length1, mulambda1 = half_strainline(x0[:,s], Tmax, h, f, xc, yc, lambdas, AB_interp,grid, pm = +1, max_notAB = max_notAB, t=t)
        line2, length2, mulambda2 = half_strainline(x0[:,s], Tmax, h, f, xc, yc, lambdas, AB_interp,grid, pm = -1, max_notAB = max_notAB, t=t)
        length = length1 + length2
        if length > 0:
            mulambda = (length1*mulambda1 + length2*mulambda2) / length
        else:
            mulambda = 0.0
        N1  = line1.shape[1]
        N2  = line2.shape[1]
        line = np.zeros((2, N1+N2-1))
        line[:,:N1] = line1[:,::-1]
        line[:,N1:] = line2[:,1:]
        lines.append(line)
        lengths.append(length)
        mulambdas.append(mulambda)
    return lines, lengths, mulambdas