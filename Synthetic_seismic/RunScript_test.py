## -------------------------------------- generate synthetic seismic in depth ------------------------------------ ##
 

    den_in_model = np.unique(mod_den)

    vel_in_model = np.unique(mod_vel)
# indice model
    MODEL = mod_den
    for i in range(0,len(exi_uni)):
        MODEL[MODEL==den_in_model[i]]=i

    units = np.zeros((len(exi_uni),2))
    for i in range(0,len(exi_uni)):
        units[i,:]=[vel_in_model[i],den_in_model[i]]  # Vp, rho


    ref = units[MODEL] # form a 3D matrix with the width1=velocity and width2 = density
    imp = np.apply_along_axis(np.product, -1, ref) # multiply velocity sand density=AI

    rc =  (imp[1:,:] - imp[:-1,:]) / (imp[1:,:] + imp[:-1,:])

    w = bruges.filters.ricker(duration=0.01, dt=0.001, f=200)

    I_1 = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                                axis=0,
                                arr=rc)
    I_2 = np.zeros((1,nx))
    I=np.concatenate((I_1,I_2), axis=0)
    fig135=plt.figure(figsize=(20,15))
    ax =plt.gca()
    plt.imshow(I, cmap=plt.get_cmap('gray'), aspect='auto',extent=[xmin,xmax,xmax,xmin])
    cbar = plt.colorbar(shrink=0.9)
    cbar.set_label('amplitudes(m/s)', rotation=90)
    plt.title('depth section convolved',fontsize=20)
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('Depth [m]', fontsize=18)



## --------------------------------------------- Convert Depth-to-time -------------------------------------------- ##

    t_max = xmax / vel_ave
 
    smoothed_V = gaussian_filter(mod_vel, sigma=sigmav) 



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
    






## --------------------------------------------- Convert time-to-Depth -------------------------------------------- ##  

  
    Vel_time = Vn 
    Synth_time = D

    tic()
    Vel_depth, synth_depth = ray_model_conv(Vel_time, Synth_time, x, t, z)
    toc()




    fig139=plt.figure(figsize=(20,15))
    ax =plt.gca()
    plt.imshow(Vel_depth, aspect='auto',extent=[xmin,xmax,xmax,xmin])
    cbar = plt.colorbar(shrink=0.9)
    cbar.set_label('velocity (m/s)', rotation=90)
    plt.title('Velocity section ---depth-to-time-converted', fontsize=20)
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('Time [s]', fontsize=18)


    fig140=plt.figure(figsize=(20,15))
    ax =plt.gca()
    plt.imshow(synth_depth, cmap=plt.get_cmap('gray'), aspect='auto',extent=[xmin,xmax,xmax,xmin])
    cbar = plt.colorbar(shrink=0.9)
    cbar.set_label('amplitudes(m/s)', rotation=90)
    plt.title('time section ---depth-to-time-converted', fontsize=20)
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('Time [s]', fontsize=18) 


t1=time.time()
print(j,"total elapsed time",t1-t0)