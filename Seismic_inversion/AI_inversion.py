# -*- coding: utf-8 -*-
"""
Boulia Post-stack inversion - 15GA_CF3

@author: 22649517
"""
import os
import sys
import segyio
import pylops
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from pylops.utils.wavelets import ricker
from scipy.interpolate import RegularGridInterpolator
from pylops.basicoperators import *
from pylops.avo.poststack import *
from pyproximal.proximal import *
from pyproximal import ProxOperator
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from depth_to_time_utils import derive, ray_image_conv, ray_model_conv

################################ FUNCTIONS ####################################
plt.close('all')
np.random.seed(10)

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
    
    
# Interpolation function
def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
    return interpolating_function((xv, yv))  

################################ Manual Setting ####################################

itmin = 0
tmax = 7000
# cdp_min = 9703
cdp_min = 10192
cdp_max = 16025
terrane_cor = 95
xmin_lim = 317297
xmax_lim = 376914
zmin_lim = 0
zmax_lim = 20000
BhLoc_1 = 326801.708
BhLoc_2 = 344697.926
BhLoc_3 = 361843.829


# Load SegY datasets - filter to 2 second between ROI cdps
segyfile = 'C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Boulia_8s_ROI_trimmed_1.sgy'
f = segyio.open(segyfile, ignore_geometry=True)

traces = segyio.collect(f.trace)[:]
ntraces, nt = traces.shape
# traces = traces[:, terrane_cor:int(np.floor(nt/2))]
traces = traces[488:, terrane_cor:3501]
ntraces, nt = traces.shape

t = f.samples[itmin:ntraces]
dt = t[1] - t[0]

plt.figure(figsize=(12, 6))
plt.imshow(traces.T, cmap='RdYlBu', vmin=-1.5, vmax=1.5, 
           extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Seismic section - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight')

plt.figure(figsize=(12, 6))
plt.imshow(traces.T, cmap='RdYlBu', vmin=-1.5, vmax=1.5, 
           extent=(xmin_lim, xmax_lim, tmax, t[0]))
plt.title('Seismic section - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' Easting')
plt.ylabel(' time (ms) ')
plt.axis('tight')



#We read also the migration velocity model. In this case, the SEG-Y file is in a regular grid, but the grid is different from that of the data
contents = sio.loadmat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/velocity_cf3_7s')
vel_mig = contents['velocities_CF3_m']
n_z, n_x = vel_mig.shape
vel_mig = vel_mig[:,:]
ntt, ntrc = traces.T.shape
vel_mig_regrid = regrid(vel_mig, ntt, ntrc)

# Display velocity
plt.figure(figsize=(12, 6))
plt.imshow(vel_mig, cmap='terrain',
           extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Vp section - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight')

# Display velocity
plt.figure(figsize=(12, 6))
plt.imshow(vel_mig, cmap='terrain',
           extent=(xmin_lim, xmax_lim, tmax, t[0]))
plt.title('Vp section - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' Easting')
plt.ylabel(' time (ms) ')
plt.axis('tight')



 
# Display data
plt.figure(figsize=(12, 6))
plt.imshow(traces.T, cmap='seismic', vmin=-6, vmax=6,
           extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Data - Seismic section - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight')

# Display regrided velocity
plt.figure(figsize=(12, 6))
plt.imshow(vel_mig_regrid, cmap='terrain',
           extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Vp regrided section  - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight')

######################################################### Convert to Depth Domain ############################################################

Vel_time = vel_mig_regrid 
Synth_time = traces.T

sio.savemat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Conversion/Vel_time_7s.mat', mdict={'Vel_time': Vel_time })
sio.savemat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Conversion/Synth_time_7s.mat', mdict={'Synth_time': Synth_time })

# did that in matlab C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Conversion/

contents = sio.loadmat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Conversion/Vel_depth_7s')
Vel_depth = contents['V_r']


contents = sio.loadmat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Conversion/seis_depth_7s')
seis_depth = contents['I_r']



fig139=plt.figure(figsize=(12,6))
ax =plt.gca()
plt.imshow(Vel_depth, cmap='terrain', aspect='auto',extent=[xmin_lim,xmax_lim,zmax_lim,0])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('velocity (m/s)', rotation=90)
plt.title('Velocity section ---time to depth-converted', fontsize=20)
plt.xlabel('Distance [m]', fontsize=18)
plt.ylabel('Depth [m]', fontsize=18)


        
fig140=plt.figure(figsize=(12,6))
ax =plt.gca()
plt.plot(BhLoc_1, 0, marker='|', color="red", markersize=100)
plt.plot(BhLoc_1, 0, marker='.', color="red", markersize=20)
plt.plot(BhLoc_2, 0, marker='|', color="blue",markersize=100)
plt.plot(BhLoc_2, 0, marker='.', color="blue",markersize=20)
plt.plot(BhLoc_3, 0, marker='|', color="green", markersize=100)
plt.plot(BhLoc_3, 0, marker='.', color="green", markersize=20)
plt.imshow(seis_depth, cmap='RdYlBu', vmin=-1.5, vmax=1.5, aspect='auto',extent=[xmin_lim,xmax_lim,zmax_lim,0])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('amplitudes(m/s)', rotation=90)
plt.xlabel('Distance [m]', fontsize=18)
plt.ylabel('Depth [m]', fontsize=18)   
ax.text(326000, 0, 'MSD001', style='italic')
ax.text(344000, 0, 'MSD002', style='italic')
ax.text(361000, 0, 'MSD003', style='italic')
      
###################################################### Read petrophyical datasets #########################################################

# 
df_1 = pd.read_csv('C:/Users/22649517/Desktop/Data/Mount_Isa/Area Info/Physical_properties_1.csv')
df_2 = pd.read_csv('C:/Users/22649517/Desktop/Data/Mount_Isa/Area Info/Physical_properties_2.csv')
df_3 = pd.read_csv('C:/Users/22649517/Desktop/Data/Mount_Isa/Area Info/Physical_properties_3.csv')

# df_1.plot('Density','Depth')
# df_1.plot(subplots=True, figsize=(5,15))
# fig = plt.subplots(figsize=(5,10))
# # set up the plot axis
# ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1, colspan=1)



fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,12))
ax1 = df_1.plot(x="Density", y="Depth",ax=axes[0],color = "red")
ax1.set_xlabel("Density (gr/cm^3)")
ax1.set_ylabel("Depth (m)")
ax1.set_title("MSD001")
ax1.set_xlim(2.2, 3.2)
ax1.set_ylim(1250, 148)
ax1.get_legend().remove()
# ax1.grid()


ax2 = df_2.plot(x="Density", y="Depth",ax=axes[1],color = "blue")
ax2.set_xlabel("Density (gr/cm^3)")
ax2.set_title("MSD002")
ax2.set_xlim(2.2, 3.2)
ax2.set_ylim(1250, 148)
ax2.get_legend().remove()
ax2.set_yticks([])
# ax2.grid()

ax3 = df_3.plot(x="Density", y="Depth",ax=axes[2],color = "green")
ax3.set_xlabel("Density (gr/cm^3)")
ax3.set_xlim(2.2, 3.2)
ax3.set_ylim(1250, 148)
ax3.set_title("MSD003")
ax3.get_legend().remove()
ax3.set_yticks([])
# ax3.grid()


################################################ Plot logs on the seismic section ######################################################
x1 = df_1.loc[:,"Density"]
x1 = x1 * 1000
y1 = df_1.loc[:,"Depth"]
x2 = df_2.loc[:,"Density"]
x2 = x2 * 1000
y2 = df_2.loc[:,"Depth"]
x3 = df_3.loc[:,"Density"]
x3 = x3 * 1000
y3 = df_3.loc[:,"Depth"]



fig140=plt.figure(figsize=(12,6))
ax =plt.gca()
plt.plot(x1 + BhLoc_1, y1,  color="red", linewidth=0.5, markersize=100)
plt.plot(BhLoc_1, 0, marker='.', color="red", markersize=10)
plt.plot(x2 + BhLoc_2, y2, color="blue",linewidth=0.5, markersize=100)
plt.plot(BhLoc_2, 0, marker='.', color="blue",markersize=10)
plt.plot(x3 + BhLoc_3, y3, color="green",linewidth=0.5, markersize=100)
plt.plot(BhLoc_3, 0, marker='.', color="green", markersize=10)
plt.imshow(seis_depth, cmap='RdYlBu', vmin=-1.5, vmax=1.5, aspect='auto',extent=[xmin_lim,xmax_lim,zmax_lim,0])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('amplitudes(m/s)', rotation=90)
plt.xlabel('Distance [m]', fontsize=18)
plt.ylabel('Depth [m]', fontsize=18)   
ax.text(326000, 0, 'MSD001', style='italic')
ax.text(344000, 0, 'MSD002', style='italic')
ax.text(361000, 0, 'MSD003', style='italic')



###################################################### generate regrided Density section #########################################################

# in Matlab 'C:\Users\22649517\Desktop\Data\Mount_Isa\seismic\FILE_GENERATOR.m
# Option: Import density model
contents = sio.loadmat('C:/Users/22649517/Desktop/Data/Mount_Isa/seismic/Density_cf3_2s')
rho_conv = contents['rho']
rho_conv = regrid(rho_conv, ntt, ntrc)
rho_conv = rho_conv * 1e3
np.unique(rho_conv)
fig140=plt.figure(figsize=(12,6))
plt.imshow(rho_conv, cmap='terrain',aspect='auto',extent=[xmin_lim,xmax_lim,zmax_lim,zmin_lim])
cbar = plt.colorbar(shrink=0.9)
cbar.set_label('Densities(kg/m^3)', rotation=90)
plt.xlabel('Distance [m]', fontsize=18)
plt.ylabel('Depth [m]', fontsize=18)   
plt.text(326000, zmin_lim, 'MSD001', style='italic')
plt.text(344000, zmin_lim, 'MSD002', style='italic')
plt.text(361000, zmin_lim, 'MSD003', style='italic')

###################################################### Compare and correlate density section with Logs #########################################################

# Density section from regional empirical equations
rho_conv = rho_conv / 1e3

loc1 = int(np.round(np.floor(326802 - xmin_lim) / 9.43))
loc2 = int(np.round(np.floor(344698 - xmin_lim) / 9.43))
loc3 = int(np.round(np.floor(361844 - xmin_lim) / 9.43))
yloc = np.linspace(zmin_lim,1250,172)

den_data = {'Depth': list(yloc),
            'Density_1': list(rho_conv[zmin_lim:172,loc1]),
            'Density_2': list(rho_conv[zmin_lim:172,loc2]),
            'Density_3': list(rho_conv[zmin_lim:172,loc3])}
den_data = pd.DataFrame(den_data)
print(den_data)

fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,12))
ax1 = df_1.plot(x="Density", y="Depth",ax=axes[0],color = "red")
ax1 = den_data.plot(x="Density_1", y="Depth",ax=axes[0],color = "black")
ax1.set_xlabel("Density (gr/cm^3)")
ax1.set_ylabel("Depth (m)")
ax1.set_title("MSD001")
ax1.set_xlim(2, 3.2)
ax1.set_ylim(1250, 0)
ax1.legend(['Log', 'Section'])
# ax1.get_legend().remove()
# ax1.grid()

ax2 = df_2.plot(x="Density", y="Depth",ax=axes[1],color = "blue")
ax2 = den_data.plot(x="Density_2", y="Depth",ax=axes[1],color = "black")
ax2.set_xlabel("Density (gr/cm^3)")
ax2.set_title("MSD002")
ax2.set_xlim(2, 3.2)
ax2.set_ylim(1250, 0)
# ax2.get_legend().remove()
ax2.set_yticks([])
ax2.legend(['Log', 'Section'])
# ax2.grid()

ax3 = df_3.plot(x="Density", y="Depth",ax=axes[2],color = "green")
ax3 = den_data.plot(x="Density_3", y="Depth",ax=axes[2],color = "black")
ax3.set_xlabel("Density (gr/cm^3)")
ax3.set_xlim(2, 3.2)
ax3.set_ylim(1250, 0)
ax3.set_title("MSD003")
# ax3.get_legend().remove()
ax3.legend(['Log', 'Section'])
ax3.set_yticks([])
# ax3.grid()

###################################################### generate regrided AI section #########################################################

aiinterp = rho_conv * vel_mig_regrid
plt.figure(figsize=(12, 6))
plt.imshow(aiinterp, cmap='terrain',
            extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('AI regrided section  - (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight');
#---------------------------------------------Statistical wavelet estimation-------------------------------------------#

nt_wav = 25 # number of samples of statistical wavelet
nfft =500  # number of samples of FFT

t_wav = np.arange(nt_wav) * (dt/1000)
t_wav = np.concatenate((np.flipud(-t_wav[1:]), t_wav), axis=0)

# Estimate wavelet spectrum
wav_est_fft = np.mean(np.abs(np.fft.fft(traces[::2], nfft, axis=-1)), axis=0)
fwest = np.fft.fftfreq(nfft, d=dt/1000)

# Create wavelet in time
wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
wav_est = wav_est / wav_est.max()

# Display wavelet
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
fig.suptitle('Estimated wavelet from seismic dataset')
axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
axs[0].set_title('Frequency')
axs[1].plot(t_wav, wav_est, 'k')
axs[1].set_title('Time')

# ------------------------------------------  2D Inversion (1) ---------------------------------------------#

# Swap time axis back to first dimension
d2d = traces.T
aiinterp2d = aiinterp

m0 = np.log(aiinterp2d)
m0[np.isnan(m0)] = 0


# Inversion with least-squares solver
epsI_b = 1e-4 # damping
niter_b = 50 # number of iterations of lsqr
wav_amp = 1e1 # guessed as we have estimated the wavelet statistically
epsR_b = 0.1 # spatial regularization
mls, rls = \
    pylops.avo.poststack.PoststackInversion(d2d, wav_amp*wav_est, m0=m0, explicit=False, 
                                            epsR=epsR_b,
                                            **dict(show=True, iter_lim=niter_b, damp=epsI_b))
mls = np.exp(mls).T

# Visualize - Inverted model
plt.figure(figsize=(12, 6))
plt.imshow(mls.T, cmap='terrain',
            extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Inverted AI model  - Least-square solver (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight');



# Visualize - all
fig, axs = plt.subplots(5, 1, figsize=(12, 25))
fig.suptitle('Least-squares inversion',
             y=0.91, fontweight='bold', fontsize=18)
axs[0].imshow(d2d, cmap='seismic', vmin=-6, vmax=6,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[0].set_title('Seismic data')
axs[0].axis('tight')
axs[1].imshow(rls, cmap='seismic', vmin=-10, vmax=10,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[1].set_title('Residual')
axs[1].axis('tight')
axs[2].imshow(aiinterp2d, cmap='terrain',
              vmin=5535, vmax=8955,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[2].set_title('Background AI')
axs[2].axis('tight')
axs[3].imshow(mls.T, cmap='terrain',
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[3].set_title('Inverted AI')
axs[3].axis('tight');
axs[4].imshow(mls.T - aiinterp2d, cmap='seismic',
              vmin=-0.7*(mls.T-aiinterp2d).max(), vmax=0.7*(mls.T-aiinterp2d).max(),
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[4].set_title('Inversion dAI')
axs[4].axis('tight');


# ------------------------------------------  Inversion with Split-Bregman ---------------------------------------------#

epsRL1_b = 1. # blocky regularization
mu_b = 1e-1 # damping for data term

msb, rsb = \
    pylops.avo.poststack.PoststackInversion(d2d, wav_amp*wav_est, m0=m0, explicit=False, 
                                            epsR=epsR_b, epsRL1=epsRL1_b,
                                            **dict(mu=mu_b, niter_outer=10, 
                                                   niter_inner=1, show=True,
                                                   iter_lim=40, damp=epsI_b))
msb = np.exp(msb).T




# Visualize - Inverted model
plt.figure(figsize=(12, 6))
plt.imshow(msb.T, cmap='terrain',
            extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Inverted AI model  - Split-Bregman (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight');



# Visualize - all
fig, axs = plt.subplots(5, 1, figsize=(12, 25))
fig.suptitle('Split-Bregman inversion',
             y=0.91, fontweight='bold', fontsize=18)
axs[0].imshow(d2d, cmap='seismic', vmin=-6, vmax=6,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[0].set_title('Seismic data')
axs[0].axis('tight')
axs[1].imshow(rsb, cmap='seismic', vmin=-10, vmax=10,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[1].set_title('Residual')
axs[1].axis('tight')
axs[2].imshow(aiinterp2d, cmap='terrain',
              vmin=5535, vmax=8955,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[2].set_title('Background AI')
axs[2].axis('tight')
axs[3].imshow(msb.T, cmap='terrain',
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[3].set_title('Inverted AI')
axs[3].axis('tight');
axs[4].imshow(msb.T - aiinterp2d, cmap='seismic',
              vmin=-0.7*(mls.T-aiinterp2d).max(), vmax=0.7*(mls.T-aiinterp2d).max(),
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[4].set_title('Inversion dAI')
axs[4].axis('tight');



# ------------------------------------------  Inversion with Primal-Dual ---------------------------------------------#



# Modelling operator
Lop = PoststackLinearModelling(wav_amp*wav_est, nt0=nt, spatdims=ntraces)
l2 = L2(Op=Lop, b=d2d.ravel(), niter=70, warm=True)

# Regularization
sigma = 0.5
l1 = L21(ndim=2, sigma=sigma)
Dop = Gradient(dims=(nt,ntraces), edge=True, dtype=Lop.dtype, kind='forward')

# Steps 
L = 8. # np.real((Dop.H*Dop).eigs(neigs=1, which='LM')[0])
tau = 1. / np.sqrt(L)
mu = 0.99 / (tau * L)

# Inversion
mpd = PrimalDual(l2, l1, Dop, m0.ravel(), tau=tau, mu=mu, theta=1., niter=50, show=True)
rpd = d2d.ravel() - Lop * mpd

mpd = np.exp(mpd).T
mpd = mpd.reshape(aiinterp2d.shape)
rpd = rpd.reshape(d2d.shape)


# Visualize - Inverted model
plt.figure(figsize=(12, 6))
plt.imshow(mpd, cmap='terrain',
            extent=(cdp_min, cdp_max, tmax, t[0]))
plt.title('Inverted AI model  - Primal-Dual (15GA-CF3, Mount Isa)')
plt.colorbar()
plt.xlabel(' CDP number')
plt.ylabel(' time (ms) ')
plt.axis('tight');



# Visualize - all
fig, axs = plt.subplots(5, 1, figsize=(12, 25))
fig.suptitle('Primal-Dual inversion',
             y=0.91, fontweight='bold', fontsize=18)
axs[0].imshow(d2d, cmap='seismic', vmin=-6, vmax=6,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[0].set_title('Seismic data')
axs[0].axis('tight')
axs[1].imshow(rpd, cmap='seismic', vmin=-10, vmax=10,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[1].set_title('Residual')
axs[1].axis('tight')
axs[2].imshow(aiinterp2d, cmap='terrain',
              vmin=5535, vmax=8955,
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[2].set_title('Background AI')
axs[2].axis('tight')
axs[3].imshow(mpd, cmap='terrain',
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[3].set_title('Inverted AI')
axs[3].axis('tight');
axs[4].imshow(mpd - aiinterp2d, cmap='seismic',
              extent=(cdp_min, cdp_max, tmax, t[0]))
axs[4].set_title('Inversion dAI')
axs[4].axis('tight');


# visualize a single trace for three inversion
t = np.linspace(0,tmax,num=910)
nxl = 6322
plt.figure(figsize=(3, 12))
plt.plot(aiinterp[:,nxl//2], t, 'k', lw=2, label='back')
plt.plot(mls[nxl//2], t, 'r', lw=2, label='LS')
plt.plot(msb[nxl//2], t, 'g', lw=2, label='Split-Bregman')
plt.plot(mpd[:, nxl//2], t, 'b', lw=2, label='PD')
plt.gca().invert_yaxis()
plt.legend();