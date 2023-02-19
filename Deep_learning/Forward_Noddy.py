
"""
@author: mahtab.rashidifard@research.uwa.edu.au
UWA
"""

## ------------------------------------ Import libraries ----------------------------------- ## 

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import gzip
from bresenham import bresenham
import scipy.io as sio
import matplotlib.pylab as plt
import numpy as np
import matplotlib as mpl
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
import time
from depth_to_time_utils import derive, ray_image_conv, ray_model_conv
import bruges  
from scipy.ndimage import gaussian_filter
import shutil
import re
import scipy.io as sio
import pandas as pd
import sys
from scipy.interpolate import RegularGridInterpolator

test_data_min=int(sys.argv[1])
test_data_max=test_data_min+100000

print(test_data_min,test_data_max)
t0=time.time()

def TicTocGenerator():
    ti = 0           
    tf = time.time() 
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti 
TicToc = TicTocGenerator() 
def toc(tempBool=True):
    ttt=0
    #tempTimeInterval = next(TicToc)
    #if tempBool:
       # print( "Elapsed time: %f seconds.\n" %tempTimeInterval)
def tic():
    ttt=0
    #toc(False)
tic()
## ---------------------------------------- Figures setting --------------------------------------- ## 

mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18
mpl.rcParams['axes.labelsize']=18
mpl.rcParams['axes.titlesize']=20

## ----------------------------------------- Manual setting ---------------------------------------- ##
 
#model
xmin=0
xmax=4000         # same for other dimensions
nx=200
ny=200
nz=200
d_mesh_size = 20
incre = 100       # constant to generate velocties from densities
# section endings
x_min = 0
x_max = 4000
y_min = 0
y_max = 4000
#samp_rate = 3    # time samples ms
sigmav = 25       # smoothing factor
plt_vis = 1000      # which plot to visualize
# Read Excell document
all_models=pd.read_csv('/Noddy/models.csv')
# k = all_models.shape[0]
# t_maxes = []

## ------------------------------------ Read g00.gz (index) file ----------------------------------- ## 
def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
    return interpolating_function((xv, yv))  
    
    
    
    
for j in range(test_data_min,test_data_max):
    root = all_models.iloc[j,1]
    path = '//'
    my_path = '//clus_1/'
    path_1 = path+root+'.g12.gz'
    path_2 = path+root+'.g00.gz'
    
    
    stream = gzip.open(path_1, 'r')
    data=np.loadtxt(stream,skiprows=0)

    model3d = data.reshape(nz,nx,ny)
    model3d2=np.transpose(model3d,(0,2,1))
    model3d2=model3d2.astype(int)
    vmin=np.amin(model3d2)
    vmax=np.amax(model3d2)

    if(j%plt_vis==0):
        fig130=plt.figure(130,figsize=(15,15))
        plt.subplot(2,2,1)
        plt.imshow(model3d2[0,:,:],extent=[xmin,xmax,xmin,xmax],vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(shrink=0.9) 
        cbar.set_label('ind', rotation=90)
        plt.title('surface')
        plt.xlabel('Horizontal position (cell index)')

        plt.subplot(2,2,2)
        plt.title('100 cells from surface')
        plt.imshow(model3d2[100,:,:],extent=[xmin,xmax,xmin,xmax],vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('ind', rotation=90)
        plt.gca().invert_yaxis()


        plt.subplot(2,2,3)
        plt.imshow(model3d2[:,0,:],extent=[xmin,xmax,xmax,xmin],vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(shrink=0.9) 
        cbar.set_label('ind', rotation=90) 
        plt.title('along x-dir, 0th cell')

    
        plt.subplot(2,2,4) 
        plt.imshow(model3d2[:,:,0],extent=[xmin,xmax,xmax,xmin],vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(shrink=0.9) 
        cbar.set_label('ind', rotation=90)
        plt.title('along y-dir, 0th cell')
        

## -------------------------------------- Choosen section --------------------------------------- ##

    xmin_ind = x_min/d_mesh_size
    xmax_ind = x_max/d_mesh_size

    ymin_ind = y_min/d_mesh_size
    ymax_ind = y_max/d_mesh_size

    x,y = bresenham(xmin_ind,ymin_ind,xmax_ind,ymax_ind)
    x=x.astype(int)
    y=y.astype(int)

    model2d = model3d2[x,y,:]

    if(j%plt_vis==0):
        fig131=plt.figure(131,figsize=(15,15))
        plt.imshow(model2d, extent=[xmin,xmax,xmax,xmin], vmin=vmin,vmax=vmax)
        cbar = plt.colorbar(shrink=0.9) 
        cbar.set_label('ind', rotation=90)
        plt.title('selected section')
        plt.xlabel('Horizontal position m ')


## --------------------------------- Extract densities from files ------------------------------ ##

    with gzip.open(path_2,'rb') as f_in:
        with open(my_path+'den_'+str(j),'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open(my_path+'den_'+str(j),'r') as f:
        x=f.readlines()[417]
    Num_of_units = [int(s) for s in x.split() if s.isdigit()][0]

    densities = []
    for i in range(419,417+1+(Num_of_units*3-1),3):
        with open(my_path+'den_'+str(j),'r') as f:
            den=f.readlines()[i]
            den_val = float(re.findall("\d+\.\d+",den)[0])
            densities.append(den_val)
    densities = np.array(densities) * 1e3 # convert to kg/m^3

## --------------------------------- Assigned extracted densities to the model ------------------------------ ##        

      
    # exi_uni = np.unique(model2d)
    # density model 
    mod_den = np.zeros((nx, ny)) 
    for o in range(0, nx):
        for p in range(0, ny):
            mod_den[o, p] = densities[model2d[o, p] - 1]
    # for i in range(0,len(exi_uni)):
    #   mod_den[mod_den==exi_uni[i]]=float(densities[exi_uni[i]-1])
    
    dmin=np.amin(mod_den)
    dmax=np.amax(mod_den)     

# velocity model 
    mod_vel = mod_den + incre
    velocities = densities + incre
    vvmin=np.amin(mod_vel)
    vvmax=np.amax(mod_vel)
    vel_ave = np.mean(mod_vel)
    
   

    if(j%plt_vis==0):
        fig133=plt.figure(133,figsize=(15,15));
        plt.imshow(mod_vel, extent=[xmin,xmax,xmax,xmin], vmin=vvmin,vmax=vvmax)
        cbar = plt.colorbar(shrink=0.9) 
        cbar.set_label('velocities (m/s) ', rotation=90)
        plt.title('selected section')
        plt.xlabel('Horizontal position (m)')
  
## -------------------------------------- generate synthetic seismic in depth ------------------------------------ ##
 

    den_in_model = np.unique(mod_den)

    vel_in_model = np.unique(mod_vel)
# indice model
    MODEL = mod_den
    for i in range(0,Num_of_units):
        MODEL[MODEL==densities[i]]=i
    MODEL=MODEL.astype(int)
    units = np.zeros((Num_of_units,2))
    for i in range(0,Num_of_units):
        units[i,:]=[velocities[i],densities[i]]  # Vp, rho


    ref = units[MODEL] # form a 3D matrix with the width1=velocity and width2 = density
    imp = np.apply_along_axis(np.product, -1, ref) # multiply velocity and density=AI

    rc =  (imp[1:,:] - imp[:-1,:]) / (imp[1:,:] + imp[:-1,:])

    w = bruges.filters.ricker(duration=0.01, dt=0.001, f=200)

    I_1 = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                                axis=0,
                                arr=rc)
    I_2 = np.zeros((1,nx))
    I=np.concatenate((I_1,I_2), axis=0)
    
    I_new = regrid(I, 480, 200)

    
    if(j%plt_vis==0):
        fig135=plt.figure(figsize=(20,15))
        ax =plt.gca()
        plt.imshow(I, cmap=plt.get_cmap('gray'), aspect='auto',extent=[xmin,xmax,xmax,xmin])
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('amplitudes(m/s)', rotation=90)
        plt.title('depth section convolved',fontsize=20)
        plt.xlabel('Distance [m]', fontsize=18)
        plt.ylabel('Depth [m]', fontsize=18)
        plt.clf()

        
        
        
## --------------------------------------------- Convert Depth-to-time -------------------------------------------- ##

    t_max = xmax / vel_ave
    #t_maxes.append(t_max)
    #nt = int(np.floor((t_max*1000)/samp_rate))
    nt = 480
    smoothed_V = gaussian_filter(mod_vel, sigma=sigmav) 
    
    
    if(j%plt_vis==0):
        fig136=plt.figure(figsize=(20,15)) 
        ax =plt.gca()
        plt.imshow(smoothed_V, aspect='auto', extent=[xmin,xmax,xmax,xmin]) 
        plt.imshow
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('velocity (m/s)', rotation=90)
        plt.title('Depth-Velocity - smoothed', fontsize=20)
        plt.xlabel('Distance [m]', fontsize=18)
        plt.ylabel('Depth [m]', fontsize=18)
        plt.grid(color='w', linestyle='--', linewidth=1)
        

# 
    x = np.linspace(0,xmax,num = nx)
    x = x * np.ones((1,nx))
    t = np.linspace(0,t_max,num = nt)
    t = t * np.ones((1,nt))
    z = np.linspace(0,xmax,num=nx);
    z = z * np.ones((1,nx))
    
# De-migration
    tic()
    Vn, D = ray_image_conv(smoothed_V, I, x, z, t)
    toc()


    
    
    if(j%plt_vis==0):
        fig137=plt.figure(figsize=(20,15))
        ax =plt.gca()
        plt.imshow(Vn, aspect='auto',extent=[xmin,xmax,t_max,xmin])
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('velocity (m/s)', rotation=90)
        plt.title('Velocity section ---depth-to-time-converted', fontsize=20)
        plt.xlabel('Distance [m]', fontsize=18)
        plt.ylabel('Time [s]', fontsize=18)
        

        fig138=plt.figure(figsize=(20,15))
        ax =plt.gca()
        plt.imshow(D, cmap=plt.get_cmap('gray'), aspect='auto',extent=[xmin,xmax,t_max,xmin])
        cbar = plt.colorbar(shrink=0.9)
        cbar.set_label('amplitudes(m/s)', rotation=90)
        plt.title('time section ---depth-to-time-converted', fontsize=20)
        plt.xlabel('Distance [m]', fontsize=18)
        plt.ylabel('Time [s]', fontsize=18)
        fig138.savefig('//synth_time_'+str(j)+'.jpg',dpi=150)
        plt.clf()





## --------------------------------------------- Convert time-to-Depth -------------------------------------------- ##  





