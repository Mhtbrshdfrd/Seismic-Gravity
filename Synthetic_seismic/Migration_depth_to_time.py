# run script for demigration (given the seismic data and velocity model in depth, it should return seismic data and vel model in time+)

import scipy.io as sio
import matplotlib.pylab as plt
import numpy as np
import matplotlib as mpl
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
import time
from depth_to_time_utils import derive, ray_image_conv



def TicTocGenerator():
    # Generator that returns time differences
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


# ----------------------------------------- Load Data---------------------------------------------- #
#   Marmousi

# =============================================================================
# contents = sio.loadmat('C:/Users/22649517/Desktop/seismic_python_finctions/input_data')
# I = contents['I'] 
# t = contents['t']
# x = contents['x']
# z = contents['z']
# V = contents['V'] 
# 
# x_min = 0
# x_max = 13000 
# z_min = 0
# z_max = 2000
# t_min = 0
# t_max = 1.4
# # =============================================================================

#   Psudo Yamarna
# =============================================================================
 
# contents = sio.loadmat('/Users//seismic_data')
# I = contents['I']
# contents = sio.loadmat('/Users/vectors')
# t = contents['t']
# x = contents['x']
# z = contents['z']
# contents = sio.loadmat('/Users//VelMod_original')
# V = contents['V']
# 
# contents = sio.loadmat('/Users//smoothed_v_fromMat')
# smoothed_V = contents['Vs']
# 
# 
# x_min = 0
# x_max = 2000
# z_min = 0
# z_max = 2000
# t_min = 0
# t_max = 0.7
# =============================================================================

# ============================================================================
    
# Synthetic 2 - highres_quick
contents = sio.loadmat('/Users//seismic_data')
I = contents['I']
contents = sio.loadmat('/Users//vectors')
t = contents['t']
x = contents['x']
z = contents['z']
contents = sio.loadmat('/Users//VelMod_original')
V = contents['V']
 
contents = sio.loadmat('/Users//smoothed_v_fromMat')
smoothed_V = contents['Vs']

x_min = 0
x_max = 2000
z_min = 0
z_max = 3000
t_min =0
t_max = 1.05
    

mpl.rcParams['xtick.labelsize']=40
mpl.rcParams['ytick.labelsize']=40
mpl.rcParams['axes.labelsize']=40
mpl.rcParams['axes.titlesize']=40

#see initial model used
fig56=plt.figure(figsize=(50,35)) # set frame of white background, width and height in inches
ax =plt.gca()
#plt.imshow(V, extent=[x_min,x_max,z_max,z_min])
plt.imshow(V, aspect='auto', extent=[x_min,x_max,z_max,z_min]) # fit image in the frame
plt.imshow
cbar = plt.colorbar(shrink=0.9)
#ax.set_aspect('equal', 'box')
cbar.set_label('velocity (m/s)', rotation=90)
plt.title('Depth-Velocity - original', fontsize=60)
plt.xlabel('Distance [m]', fontsize=50)
plt.ylabel('Depth [m]', fontsize=50)
#ax.set_aspect('equal', 'box')
plt.grid(color='w', linestyle='--', linewidth=1)

# if synthetic data, it should be smoothed, otherwise, ignore
# 1 -- Gaussian filter
# =============================================================================
# fig57=plt.figure(figsize=(50,35))
# ax =plt.gca()
# smoothed_V = gaussian_filter(V, sigma=5)
# plt.imshow(smoothed_V, aspect='auto',extent=[x_min,x_max,z_max,z_min])
# cbar = plt.colorbar(shrink=0.9)
# cbar.set_label('velocity (m/s)', rotation=90)
# plt.title('Depth-Velocity - smoothed (Gaussian filter win = 3)', fontsize=60)
# plt.xlabel('Distance [m]', fontsize=50)
# plt.ylabel('Depth [m]', fontsize=50)
# 
# # 2-- convolution
# V_filter = V.copy()
# filter_mat = np.zeros((100,100))
# =============================================================================

# =============================================================================
# mat_filt=np.zeros((2,3,3))
# for k in range(0,3):
#     for l in range(0,3):
#         for m in range(0,3): 
#             mat_filt[k,l,m]=np.exp(-((k-1)**2+(l-1)**2+(m-1)**2+1e-5)*(1+i))
#             mat_filt=mat_filt/np.sum(mat_filt)
#             V_filt=V.copy()
#             for j in range(0,n_lset):
#                 phi_filt[j]=convolve(phi_filt[j],mat_filt,mode='same')
# =============================================================================

# call function, taking a depth_migrated image and smoothed velocity ---> creates time-migrated and Dix_velocity model
# De-migration
tic()
Vn, D = ray_image_conv(smoothed_V, I, x, z, t)
toc()

fig60=plt.figure(figsize=(50,35))
ax =plt.gca()
plt.imshow(Vn, aspect='auto',extent=[x_min,x_max,t_max,z_min])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('velocity (m/s)', rotation=90)
plt.title('Velocity section ---depth-to-time-converted', fontsize=60)
plt.xlabel('Distance [m]', fontsize=50)
plt.ylabel('Time [s]', fontsize=50)


fig60=plt.figure(figsize=(50,35))
ax =plt.gca()
plt.imshow(D, cmap=plt.get_cmap('gray'), aspect='auto',extent=[x_min,x_max,t_max,t_min])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('amplitudes(m/s)', rotation=90)
plt.title('time section ---depth-to-time-converted', fontsize=60)
plt.xlabel('Distance [m]', fontsize=50)
plt.ylabel('Time [s]', fontsize=50)







