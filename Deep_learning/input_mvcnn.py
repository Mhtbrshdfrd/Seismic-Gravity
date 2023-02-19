"""
mahtab.rashidifard@research.uwa.edu.au
UWA
"""

## ------------------------------------ Import libraries ----------------------------------- ## 

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import matplotlib as mpl
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
import time  
from scipy.ndimage import gaussian_filter
import shutil
import re
import scipy.io as sio
import pandas as pd
import sys
from scipy.interpolate import RegularGridInterpolator
import imageio as iio

## ---------------------------------------- Figures setting --------------------------------------- ## 

mpl.rcParams['xtick.labelsize']=18
mpl.rcParams['ytick.labelsize']=18
mpl.rcParams['axes.labelsize']=18
mpl.rcParams['axes.titlesize']=20

## ----------------------------------------- Manual setting ---------------------------------------- ##
 

models=pd.read_csv('/group/cet/mrashidifard/noddy/mahtab/models.csv')

fo_fo_u_df = models[models['event_all'] == 'FOLD FOLD UNCONFORMITY']
fa_fa_u_df =  models[models['event_all'] == 'FAULT FAULT UNCONFORMITY'] 
fa_fo_u_df =  models[models['event_all'] == 'FAULT FOLD UNCONFORMITY'] 
fo_fa_u_df =  models[models['event_all'] == 'FOLD FAULT UNCONFORMITY'] 

ind_fo_fo_u = fo_fo_u_df.iloc[:,0]
ind_fa_fa_u = fa_fa_u_df.iloc[:,0]
ind_fa_fo_u = fa_fo_u_df.iloc[:,0]
ind_fo_fa_u = fo_fa_u_df.iloc[:,0]


## ------------------------------------ Addressing images-1 ----------------------------------- ##  
#for i in range(ind_dyke.shape[0]): 

for i in range(12000): 
    ind_fo_fo_u_c = ind_fo_fo_u.iloc[i]
    img01 = iio.imread('/group/cet/mrashidifard/mvcnn/06_Apr_22/images/seis_images_2/seis_image_'+str(ind_fo_fo_u_c)+'.jpg')
    #img02 = iio.imread('/group/cet/mrashidifard/mvcnn/06_Apr_22/images/grav_images/grav_image_'+str(ind_fo_fo_u_c)+'.jpg')
    fig01=plt.figure(figsize=(15,15))
    plt.imshow(img01, cmap='Greys', aspect='auto') 
    plt.axis('off')
    fig01.savefig('/group/cet/mrashidifard/mvcnn/06_Apr_22/labels/fo_fo_u/train/Obj_'+str(i)+'_v03.png',bbox_inches='tight',pad_inches = 0,dpi=20)
    plt.clf()
    plt.close(fig01)
print('fo_fo_u train finished')
'''
    plt.close(fig01)
    fig02=plt.figure(figsize=(15,15))
    plt.imshow(img02, cmap='Greys', aspect='auto') 
    plt.axis('off')
    fig02.savefig('/group/cet/mrashidifard/mvcnn/06_Apr_22/labels/fo_fo_u/train/Obj_'+str(i)+'_v02.png',bbox_inches='tight',pad_inches = 0,dpi=20)
    plt.clf()
    plt.close(fig02)
'''
for i in range(3000): 
    ind_fo_fo_u_c = ind_fo_fo_u.iloc[12000+i]
    img01 = iio.imread('/group/cet/mrashidifard/mvcnn/06_Apr_22/images/seis_images_2/seis_image_'+str(ind_fo_fo_u_c)+'.jpg')
    #img02 = iio.imread('/group/cet/mrashidifard/mvcnn/06_Apr_22/images/grav_images/grav_image_'+str(ind_fo_fo_u_c)+'.jpg')
    fig01=plt.figure(figsize=(15,15))
    plt.imshow(img01, cmap='Greys', aspect='auto') 
    plt.axis('off')
    fig01.savefig('/group/cet/mrashidifard/mvcnn/06_Apr_22/labels/fo_fo_u/test/Obj_'+str(i+12000)+'_v03.png',bbox_inches='tight',pad_inches = 0,dpi=20)
    plt.clf()
    plt.close(fig01)
    '''
    fig02=plt.figure(figsize=(15,15))
    plt.imshow(img02, cmap='Greys', aspect='auto') 
    plt.axis('off')
    fig02.savefig('/group/cet/mrashidifard/mvcnn/06_Apr_22/labels/fo_fo_u/test/Obj_'+str(i+12000)+'_v02.png',bbox_inches='tight',pad_inches = 0,dpi=20)
    plt.clf()
    plt.close(fig02)
print('fo_fo_u finished')

'''
print('fo_fo_u test finished')