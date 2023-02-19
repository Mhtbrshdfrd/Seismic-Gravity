%% Simple code for straight ray-tracing of seismic travel times


%% Defining Initial models  
nx=36; nz=24;  Dx=25;Dz=25;  Ntot=nx*nz; xaxes=nx*Dx; zaxes=nz*Dz; 
Ob1=[3000 (1/3000)*10^6 2.29];    % object 1 velocity in m/s, slowness (micro_sec/meter) and density gr/cm^3
Ob2=[1000 (1/1000)*10^6 1.74];    
Bac=[2000 (1/2000)*10^6 2.07];    % Background 


Vel=Bac(1)*ones(nz,nx);     
Vel(10:15,5:12)=Ob1(1);
Vel(10:20,20:30)=Ob2(1);

Sln=Bac(2)*ones(nz,nx);
Sln(10:15,5:12)=Ob1(2);
Sln(10:20,20:30)=Ob2(2);

AI=[Bac(1)*Bac(3) Ob1(1)*Ob1(3) Ob2(1)*Ob2(3)];
Model2=AI(1)*ones(nz,nx);
Model2(10:15,5:12)=AI(2);
Model2(10:20,20:30)=AI(3);

rx=[];
for k=1:nz-1 
    for h=1:nx
      rx(k,h)=(Model2(k+1,h)-Model2(k,h))/(Model2(k+1,h)+Model2(k,h));
    end
end
ry=[];
for i=1:nx-1
    for j=1:nz
       ry(j,i)=(Model2(j,i+1)-Model2(j,i))/(Model2(j,i+1)+Model2(j,i));
    end
end
Refl=rx(:,1:nx-1)+ry(1:nz-1,:); 




%% Acquisition geometry
 

Sy=(0:10:600);
Sx=zeros(1,61);

Ry=(0:10:600);
Rx=ones(1,61).*900;
n=size(Sx,2);
raytot=size(Sx,2)*size(Rx,2);


%plot initial slowness model
figure;
x_axis = [0 xaxes];z_axis=[0 zaxes];
imagesc(x_axis,z_axis,Sln);
set(gca,'Color','w');hh=colorbar;title(hh,'slowness (microsec/m)');ylabel('Depth (m)');xlabel('Distance (m)');title('Initial slowness model')
hold on
plot(Sx,Sy,'p','LineWidth',1,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',10)
plot(Rx,Ry,'<','LineWidth',2,'MarkerEdgeColor','y','MarkerFaceColor','y','MarkerSize',5)



%% Calculating sensitivity Kernels 
fprintf('start of Kernel calculation ...\n');
x=0:10:900; % in meter
for i=1:raytot
a=ceil(i./n);
b=(i-(a-1).*n);
[p1]=polyfit([Sx(a) Rx(b)],[Sy(a) Ry(b)],1); 
for k=1:nz
for l=1:nx
if polyval(p1,x(l)) >= (k-1)*Dz && polyval(p1,x(l)) < k*Dz... 
&& x(l) >= (l-1)*Dx && x(l) < l*Dx
d(k,l,i)=distance(x(l),polyval(p1,x(l)),... 
x(l+1),polyval(p1,x(l+1))); 
t(k,l,i)=d(k,l,i)./Vel(k,l);
else
d(k,l,i)=0;
t(k,l,i)=0;
end
end
end
dt(i,:)=reshape(d(:,:,i)',1,Ntot); %final kernel matrix - in meter
tt(i,:)=reshape(t(:,:,i)',1,Ntot); 
end
fprintf('...End of Kernel calculation \n');
 
% Forward Modelling                
t_fw = sum(tt,2); 
t_obs= sqrt(0.00015)*abs(randn(size(t_fw)))+ t_fw ;  %measured time IN SEC
t_Obs=t_obs.*10^6;  % time in microsecond
misfit = abs(t_obs - t_fw);
x_axes = 1:raytot;










