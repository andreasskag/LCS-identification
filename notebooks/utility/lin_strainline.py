from scipy.interpolate import RectBivariateSpline
from numba import njit
# coding: utf-8

# In[ ]:
import numpy as np

def setNXNY(Nx_, Ny_):
    global Nx, Ny
    Nx = Nx_
    Ny = Ny_


def rk4(X, t, h, f, scale=1):
    k1 = scale*f(X,          t)
    k2 = scale*f(X + k1*h/2, t + h/2)
    k3 = scale*f(X + k2*h/2, t + h/2)
    k4 = scale*f(X + k3*h,   t + h)
    return X + h*(k1 + 2.*k2 + 2.*k3 + k4) / 6.


class LinearSpecial():
    def __init__(self, xc, yc, xi):
        self.xc = xc
        self.yc = yc
        self.dx = xc[1] - xc[0]
        self.dy = yc[1] - yc[0]
        self.xi = xi
        self.Nx = len(xc)
        self.Ny = len(yc)
        self.fold = None
        
    def __call__(self, X, t):
        # Calculate indices for lower left corner in cell
        i = np.floor((X[0] - self.xc[0]) / self.dx).astype(np.int32)
        j = np.floor((X[1] - self.yc[0]) / self.dy).astype(np.int32)
        
        # If outside the domain, stop
        if (i >= Nx - 2) or (j >= Ny - 2) or (i < 0) or (j < 0):
            raise IndexError
        
        # Use the lower left corner as reference, calculate
        # the rotation of the other vectors, and rotate by 180
        # degrees if required (due to orientational discontinuity)
        subxi = self.xi[:,i:i+2, j:j+2].copy()
        dotp = np.sum(subxi[:,0,0].reshape(2,1,1) * subxi, axis = 0)
        subxi[:, dotp < 0] =  -subxi[:, dotp < 0]
        
        # Linear interpolation
        Wx0 = (self.xc[i+1] - X[0]) / self.dx
        Wx1 = 1 - Wx0
        Wy0 = (self.yc[j+1] - X[1]) / self.dy
        Wy1 = 1 - Wy0

        V = Wy0*(Wx0*subxi[:,0,0] + Wx1*subxi[:,1,0]) + Wy1*(Wx0*subxi[:,0,1] + Wx1*subxi[:,1,1])
                
        # Check orientation against previous vector
        if self.fold is None:
            return V
        else:
            # If dot product is negative, flip direction
            return V * np.sign(np.dot(V, self.fold))
            
def half_strainline(x0, Tmax, h, f, xc, yc, lambdas, ABtrue, grid, pm, max_notAB = 0.3, t=0):
    # Re-initialise the f-function
    f.fold = None

    Nt = int((Tmax-t) / h)
    xs = np.zeros((2, Nt))
    xs[:,0] = x0
    dx = xc[1] - xc[0]
    dy = yc[1] - yc[0]
    
    # Buffer zone outside domain
    xbuf = grid[0,1,0] - grid[0,0,0]
    ybuf = grid[1,0,1] - grid[1,0,0]
    
    ##index = np.empty(2)
    ##index[0] = np.argmin(np.abs(x0[0] - xc))
    ##index[1] = np.argmin(np.abs(x0[1] - yc))
    
    #lambdas1 = RectBivariateSpline(xc, yc, lambdas[0, :, :])
    #lambdas2 = RectBivariateSpline(xc, yc, lambdas[1, :, :])
    #lamb1 = lambdas[0](x0[0], x0[1])
    #lamb2 = lambdas[1](x0[0], x0[1])
    ##lamb1 = lambdas[0, int(index[1]), int(index[0])]
    ##lamb2 = lambdas[1, int(index[1]), int(index[0])]
    
    #print(scaling)
    bound_xlow = grid[0,0,0]
    bound_xhigh = grid[0,-1,0]
    bound_ylow = grid[1,0,0]
    bound_yhigh = grid[1,0,-1]
    
    #print(bound_xlow)
    #print(bound_xhigh)
    # Parameters of strainline
    AB_interp = RectBivariateSpline(xc, yc, ABtrue)
    length = 0.0
    notABlength = 0.0
    mulambda = 0.0
    
    for n in range(1, Nt):
        #print(n)
        #f.fold = f(xs[:,n-1], t)
        lamb1 = lambdas[0](xs[0,n-1], xs[1,n-1])
        lamb2 = lambdas[1](xs[0,n-1], xs[1,n-1])
        
        ##index[0] = np.argmin(np.abs(xs[0, n-1] - xc))
        ##index[1] = np.argmin(np.abs(xs[1, n-1] - yc))
        ##lamb1 = lambdas[0, int(index[1]), int(index[0])]
        ##lamb2 = lambdas[1, int(index[1]), int(index[0])]
        scaling = (lamb1 - lamb2) / (lamb1 + lamb2)
        scaling = scaling[0,0]**2
        
        if scaling > 1: 
            scaling = 1
        elif scaling < 1e-5:
            break #stuff is singular    
        f.fold = 1*f(xs[:,n-1], t)
        
                
        try:
            xs[:,n] = rk4(xs[:,n-1], t, pm*h, f, scale = scaling)
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
        ##mulambda += lambdas[1,i,j] * dl
        mulambda += lambdas[1](xs[0,n], xs[1,n]) * dl
        # Check if A and B are satisfied
        if AB_interp(xs[0,n], xs[1,n]) > 0.45:
        #if ABtrue[i, j]:
            notABlength = 0
        else:
            notABlength += dl
            #print(dl)
            

    if length > 0:
        mulambda = mulambda / length
    else:
        mulambda = 0.0
    return xs[:,:n], length, mulambda

def strainline(x0, Tmax, h, f, xc, yc, lambdas, ABtrue, grid, max_notAB = 0.3, t=0):
    num_iters = x0.shape[1]
    lines = []
    mulambdas = []
    lengths = []
    for s in range(num_iters):
        line1, length1, mulambda1 = half_strainline(x0, Tmax, h, f, xc, yc, lambdas, ABtrue, grid, pm = +1, max_notAB = max_notAB, t=t)

        line2, length2, mulambda2 = half_strainline(x0, Tmax, h, f, xc, yc, lambdas, ABtrue, grid, pm = -1, max_notAB = max_notAB, t=t)
        length = length1 + length2
        if length > 0:
            mulambda = (length1*mulambda1 + length2*mulambda2) / length
        else:
            mulambda = 0.0
        N1  = line1.shape[1]
        N2  = line2.shape[1]
        line = np.zeros((2, N1+N2))
        line[:,N2:] = line1[:,:]
        line[:,:N2] = line2[:,::-1]
        lines.append(line)
        lengths.append(length)
        mulambdas.append(mulambda)
    return lines, lengths, mulambdas

