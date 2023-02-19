

# Create synthetic seismic (zero offset)
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib as mpl



# Synthetic 2
mpl.rc('xtick', labelsize=50)
mpl.rc('ytick', labelsize=50)
x_min = 0
x_max = 2000
z_min = 0
z_max = 3000
t_min = 0
t_max = 1.2
length, depth = 2000, 3000
contents = sio.loadmat('/Users//denMod_original_paper')
model= contents['den']
model[model==0]=2450
model[model==20]=2600
model[model==100]=2820
model[model==180]=3000
model[model==300]=3100
model[model==430]=3100
# =============================================================================
# contents = sio.loadmat('C:/Users/22649517/Desktop/Data/syntheticData/synthetic_2/Migration/VelMod_high')
# model= contents['V_high']
# =============================================================================
#plot model

fig56=plt.figure(figsize=(50,35)) # set frame of white background, width and height in inches
ax =plt.gca()
plt.imshow(model, aspect='auto', extent=[x_min,x_max,z_max,z_min]) # fit image in the frame
plt.imshow
cbar = plt.colorbar(shrink=0.9)
#ax.set_aspect('equal', 'box')
cbar.set_label('velocity (m/s)', rotation=90)
plt.title('Depth-Velocity - original', fontsize=60)
plt.xlabel('Distance [m]', fontsize=50)
plt.ylabel('Depth [m]', fontsize=50)
#ax.set_aspect('equal', 'box')
plt.grid(color='w', linestyle='--', linewidth=1)


# indexing the initial model from 0 
model[model==2450]=0
model[model==2600]=1
model[model==2820]=2
model[model==3000]=3
model[model==3100]=4
model[model==3300]=5
 # order should be based on the numbering above
rocks = np.array([[2450, 2690],# Vp, rho
                  [2600, 2770],
                  [2820, 2850],
                  [3000, 2790],
                  [3100, 3100],
                  [3300, 3300]])


# reflectivity
earth = rocks[model] # form a 3D matrix with the width1=velocity and width2 = density
imp = np.apply_along_axis(np.product, -1, earth) # multiply velocity sand density=AI

rc =  (imp[1:,:] - imp[:-1,:]) / (imp[1:,:] + imp[:-1,:])
fig50=plt.figure(figsize=(50,35))
ax =plt.gca()
plt.imshow(rc, cmap='Greys', aspect='auto', extent=[x_min,x_max,z_max,z_min])
plt.imshow
cbar = plt.colorbar(shrink=0.9)
#ax.set_aspect('equal', 'box')
cbar.set_label('reflectivity',rotation=90)
plt.title('Reflectivity',fontsize=60)
plt.xlabel('Distance [m]', fontsize=60)
plt.ylabel('Depth [m]', fontsize=60)


# convolution

import bruges  #  wavelet function from this package

w = bruges.filters.ricker(duration=0.01, dt=0.001, f=80)

plt.plot(w)
plt.show()


synth = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                            axis=0,
                            arr=rc)

fig60=plt.figure(figsize=(50,35))
ax =plt.gca()
plt.imshow(synth, cmap=plt.get_cmap('gray'), aspect='auto',extent=[x_min,x_max,t_max,t_min])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('amplitudes(m/s)', rotation=90)
plt.title('depth section convolved',fontsize=60)
plt.xlabel('Distance [m]', fontsize=60)
plt.ylabel('Time [s]', fontsize=60)


# =============================================================================
# plt.imshow(synth, cmap="Greys", aspect=0.2)
# plt.show()
# =============================================================================
 