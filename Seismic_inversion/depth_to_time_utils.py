# functions for depth to time migration
import matplotlib.pylab as plt
import matplotlib.pyplot as pplt
import numpy as np
import matplotlib as mpl
import time


def TicTocGenerator():
    # Generator that retu- rns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = np.array([10,5,8,9,15,22,26,11,15,16,18,7])


def derive(x,t):
    
    # y=x(t) , dy/dx with truncation error o(delta_t^2)
    [m,n] = np.shape(x)
    p = len(t)
    y = np.zeros((m,n))
    if p==1:
        d = 2 * t
        y[:,0] = (-3 * x[:,0] + 4 * x[:,1] - x[:,2]) / d
        y[:,1:n-2] = ( x[:,2:n-1] - x[:,0:n-3] ) / d;
        y[:,n-1] = ( 3 * x[:,n-1] - 4 * x[:,n-2] + x[:,n-3])/d
    else:
        # 2dfying t
        t = np.ones((m,1)) * t
        dt = t[:,1:n-1] - t[:,0:n-2]
        dy = (x[:,1:n-1] - x[:,0:n-2]) /dt
        y = (dy[:,0:n-3] * dt[:,1:n-2] + dy[:,1:n-2] * dt[:,0:n-3]) / ...
        (t[:,2:n-1] - t[:,0:n-3]);
        y = np.append(np.array([y[:,0]]).T,y,axis=1)
        y = np.append(y,np.array([y[:,n-3]]).T,axis=1)
     
    return y


    
## D2T Conversion
def ray_image_conv(V, I, x, z, t):
    [l, nx] = np.shape(x)
    [l, nz] = np.shape(z)
    [l, nt] = np.shape(t)

    x=x.reshape(nx,1, order='F')
    z=z.reshape(nz,1, order='F')
    t=t.reshape(nt,1, order='F')

    dx = abs(x[0]-x[1])            # cdp interval
    dz = abs(z[0]-z[1])            # sample interval in depth
    #
    dVdx = derive(V,dx)            # derivatives are computed using second order finite difference
    dVdz = derive(V.T,dz).T
    #
    dt = abs(t[0]-t[1]);           # time sample
    # velocity and time sec in time domain (output)     
    Vn = np.zeros((nt,nx));        
    D = np.zeros((nt,nx)); 
    #
    Vn[0,:] = V[0,:]    # Initial Condition
    zt = np.zeros((1,nx))
# =============================================================================
    [l, nzt] = np.shape(zt)
    zt=zt.reshape(nzt,1, order='F')
# =============================================================================
    xt = x
    # horizontal slowness  p = (partial T / Partial x)
    p = np.zeros((1,nx))
# =============================================================================
    [l, npp] = np.shape(p)
    p=p.reshape(npp,1, order='F')
# =============================================================================
    # vertical slowness ()   q = (Partial T / Partial z)
    q = 1 / V[0,:]        
    #
    D[0,:] = I[0,:]   # Initial Condition
    #

    
    
    # the way that it goes down to the section (first time then cdp)
    for it in range(1,nt):
        np.disp(it+1)
        tic()
        for ix in range(0,nx):
            une = abs(z-zt[ix])
            dux = abs(x-xt[ix])
            iz = np.where(une == une.min()) # index of minimum
            iz = iz[0]                      # as iz is a tuple
            iX = np.where(dux == dux.min()) # index of minimum, read as tuple
            iX = iX[0]
            aux_v = V[iz,iX]
            aux_d = I[iz,ix]
            #
            aux_dvx = dVdx[iz,iX]
            #
            aux_dvz = dVdz[iz,iX]
            
            # Charactristic method for Hamiltonian-Jacobi transformation
            
            zt[ix] = zt[ix] + dt * q[ix] * aux_v**2  # dz/dt = (Partial H / Partial q)
            xt[ix] = xt[ix] + dt * p[ix] * aux_v**2  # dx/dt = (Partial H / Partial p)
            #
            q[ix] = q[ix] - dt * aux_dvz / aux_v     # dq/dt (Vertical)   = -(Partial H / Partial z)
            p[ix] = p[ix] - dt * aux_dvx / aux_v     # dp/dt (horizontal) = -(Partial H / Partial z)
            #
            Vn[it,ix] = aux_v
            D[it, ix] = aux_d
        toc()
    return Vn, D   
      
  
## T2D Conversion

def ray_model_conv(Vn, D, x, t, z):
    [l, nx] = np.shape(x)
    [l, nz] = np.shape(z)
    [l, nt] = np.shape(t)
    
    
    x=x.reshape(nx,1, order='F')
    z=z.reshape(nz,1, order='F')
    t=t.reshape(nt,1, order='F')
    dx = abs(x[0]-x[1])            # cdp interval
    dt = abs(t[0]-t[1])            # sample interval in time
    
    dVdx = derive(Vn,dx)            # derivatives are computed using second order finite difference
    dVdt = derive(Vn.T,dt).T
    
    dz = abs(z[0]-z[1]);           # time sample   
    V = np.zeros((nz,nx));        
    I = np.zeros((nz,nx)); 
    
    # initial conditions
    
    V[0,:]=Vn[0,:]
    
    tz=np.zeros((1,nx))
    [l, nzt] = np.shape(tz)
    tz=tz.reshape(nzt,1, order='F')
    
    xz = x
    
    
    p = np.zeros((1,nx))
    [l, npp] = np.shape(p)
    p=p.reshape(npp,1, order='F')
    
    
    q = Vn[0,:]
    nqq = np.shape(q)
    q=q.reshape(nqq[0],1, order='F')
    
    I[0,:] = D[0,:]   # Initial Condition
    
    
    for iz in range(1,1):
        print('ololo 1',iz)

        for ix in range(0,nx):
            print('ololo 2', ix)
            
            une = abs(t-tz[ix])
            dux = abs(x-xz[ix])
            it = np.asarray(np.where(une == une.min())) # index of minimum
            # it = it[0]
            it =it[0][0] 
            # as iz is a tuple
            iX = np.asarray(np.where(dux == dux.min())) # index of minimum, read as tuple
            iX = iX[0][0]
            aux_v = Vn[it,iX]
            aux_i = D[it,ix]
            #
            aux_dvx = dVdx[it,iX]
            #
            aux_dvt = dVdt[it,iX]
            
            # Charactristic method for Hamiltonian-Jacobi transformation
            
            tz[ix] = tz[ix] + dz * q[ix]/aux_v**2  # dz/dt = (Partial H / Partial q)  ERROR MESSAGE
            xz[ix] = xz[ix] + dz * p[ix]             # dx/dt = (Partial H / Partial p)
            #
            # q[ix] = q[ix] + dz*aux_dvt / aux_v**3*q[ix]**2    # dq/dt (Vertical)   = -(Partial H / Partial z)
            # p[ix] = p[ix] + dz*aux_dvx / aux_v**3*q[ix]**2     # dp/dt (horizontal) = -(Partial H / Partial z)
            
            q[ix] = q[ix] + dz * aux_dvt / aux_v**3 * q[ix]**2    # dq/dt (Vertical)   = -(Partial H / Partial z)
            p[ix] = p[ix] + dz * aux_dvx / aux_v**3 * q[ix]**2     # dp/dt (horizontal) = -(Partial H / Partial z)
            
            #
            V[iz,ix] = aux_v
            I[iz,ix] = aux_i
        toc()
    return V, I 
    
    
    
    
    # for iz in range(1,10):
    #     np.disp(iz)
    











       