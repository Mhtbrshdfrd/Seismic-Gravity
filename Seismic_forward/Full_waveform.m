%% Simple code for Full-waveform of seismic travel times

%% Model 1
load('\Users\VelMod_original.mat','V');
vmodel = V;
nx =80; nz=80; dx = 25;
xaxes=nx*dx;zaxes=nz*dx;
x=(0:2:nx-1)*dx;z=(0:nz-1)*dx; 
nsources=8;
coef = nx/nsources;
%% Model 2 - layered cake

nx=128;dx=10;nz=128; 
nsources=8;
coef = nx/nsources;
x=(0:2:nx-1)*dx;z=(0:nz-1)*dx;
xaxes=nx*dx;zaxes=nz*dx;
v1=2000;v2=2800;v3=3200; 
vmodel=v3*ones(nx,nz); 
z1=(nz/8)*dx;z2=(nz/2)*dx;
dx2=dx/2;

%first layer
xpoly=[-dx2 max(x)+dx2 max(x)+dx2 -dx2];zpoly=[-dx2 -dx2 z1+dx2 z1+dx2];
vmodel=afd_vmodel(dx,vmodel,v1,xpoly,zpoly);

%second layer
zpoly=[z1+dx2 z1+dx2 z2+dx2 z2+dx2];
vmodel=afd_vmodel(dx,vmodel,v2,xpoly,zpoly);

rho=vmodel;
%  Interval velocity
V_1=ones(1,128); 
V_1(1:16)=v1;V_1(17:64)=v2;V_1(65:128)=v3;
Z_1=1:128;
t_2=vint2t(V_1,Z_1);
% 
vrms=vint2vrms(V_1,t_2); 
x_vrms=1:128;
%% 

dtstep=0.001; 
dt=0.002;tmax=0.698;


%% configuration - static 

xrec=x;
zrec=zeros(size(xrec));
snap1=zeros(size(vmodel));
snap2=snap1;              

xsou=zeros(1,nsources-1);
for i=1:nsources-1
    snap2(1,coef*i)=1; 
    xsou(1,i)=(coef*i)*dx;
end
zsou=zeros(size(xsou));

%place-----sources----snap 2 ---------(uncomment for single shot)----------
% snap2(1,nx/2)=1; %place the source
% xsou=(nx/2)*dx;%source coordinate for ploting
% zsou=zeros(size(xsou));
% % snap2(1,length(x)*0.75)=1; %place the source

%
figure(2)
x_axis = [0 xaxes];z_axis=[0 zaxes];t_axis=[0 tmax];
imagesc(x_axis,z_axis,vmodel);xlabel('Distance (m)');ylabel('Depth (m)');
set(gca,'Color','w');colorbar;title('initial velocity model');title('velocity in m/s')
hold on
plot(xsou,zsou,'p','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',10)
plot(xrec,zrec,'v','LineWidth',1,'MarkerEdgeColor','y','MarkerFaceColor','y','MarkerSize',3)
hold off

%% trace calculation (FW)


[seismogram2,seis2,t]=afd_shotrec(dx,dtstep,dt,tmax, ...
    vmodel,snap1,snap2,xrec,zrec,[5 10 30 40],0,2); 
plotimage(seismogram2); 



%% Split - spread acquisition geomeytry 

snap1=zeros(size(vmodel));
snap2=snap1;


for i=0:9
    snap1=zeros(size(vmodel));
    snap2=snap1;
    xsou=zeros(1,1);        
    snap2(1,i+1)=1; 

    xsou(1,1)= dx/2 + (dx * i) ; 
    zsou=zeros(size(xsou));

    xrec = dx:10:(nx*dx)-dx; 
    save('\Users\xrec.mat\','xrec');
    zrec=zeros(size(xrec));

    subplot(5,2,i+1);
    x_axis = [0 xaxes];z_axis=[0 zaxes];t_axis=[0 tmax];
    imagesc(x_axis,z_axis,vmodel);xlabel('Distance (m)','FontSize',7,'Color','k');ylabel('Depth (m)','FontSize',7,'Color','k');
    set(gca,'Color','w','FontSize',8);colorbar;title('initial velocity model');title(sprintf('SHOTID = %d',i+1),'FontSize',10,'Color','k');
     hold on
     plot(xsou,zsou,'p','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',10)
     plot(xrec,zrec,'v','LineWidth',1,'MarkerEdgeColor','y','MarkerFaceColor','y','MarkerSize',3)
     hold on
end 

%% Forward Modelling 

for i=0:9
    snap1=zeros(size(vmodel));
    snap2=snap1;
    xsou=zeros(1,1);        
    snap2(1,i+1)=1; %place the source in snap2 % source spacing==cell size

    xsou(1,1)= dx/2 + (dx * i) ;
    zsou=zeros(size(xsou));

    xrec = dx:10:(nx*dx)-dx; % receiver spacing 
    
    zrec=zeros(size(xrec));

    subplot(5,2,i+1);
    [seismogram2,seis2,t]=shotrec(dx,dtstep,dt,tmax, ...
    vmodel,snap1,snap2,xrec,zrec,[5 10 30 40],0,2); 
    Path = '\Users\SHOTGATH\';
    baseName = sprintf('SHOTID%d.mat', i+1);
    FullMatName = fullfile(Path,baseName); 
    save(FullMatName,'seismogram2');
    imagesc(seismogram2);colormap(gray);
    x_axis = [0 xaxes];z_axis=[0 zaxes];t_axis=[0 tmax];
    imagesc(x_axis,t_axis,seismogram2);xlabel('Offset (m)','FontSize',7,'Color','k');ylabel('Time (s)','FontSize',7,'Color','k');
    set(gca,'Color','w','FontSize',8);colorbar;title('Amplitude');title(sprintf('SHOTID = %d',i+1),'FontSize',10,'Color','k');
    WriteSegy(sprintf('SHOTID%d.sgy', i+1),seismogram2,'dsf',1,'revision',1);
    %add geometry to trace header
    [Data,SegyTraceHeaders,SegyHeader]=ReadSegy(sprintf('SHOTID%d.sgy', i+1));
    for j=1:length(xrec)
        CdpX = (xsou + xrec(j))/2;
        CdpY = zeros(size(xsou));
    end
    WriteSegyTraceHeaderValue(sprintf('SHOTID%d.sgy', i),CdpX,'key','cdpX');
    WriteSegyTraceHeaderValue(sprintf('SHOTID%d.sgy', i),CdpY,'key','cdpY');
    WriteSegyTraceHeaderValue(sprintf('SHOTID%d.sgy', i),xsou,'key','SourceX'); 
    WriteSegyTraceHeaderValue(sprintf('SHOTID%d.sgy', i),zsou,'key','SourceX'); 
    plotimage(seismogram2);

end 

%% Shotgather NMO and map to cmp


load('\Users\rmsvel_inTime.mat\','V_dix');
load('\Users\Image_ray_input.mat\','Vs','I','x_IR','z_IR','t_IR');
shot = seismogram2;
xshot = xsou;
Cr = movmean(V_dix,10,2); 
vrmsmod = movmean(Cr,10,1);
xv = x_IR';
tv = t_IR';
dxcmp = 5; 
x0cmp = 6.25; 
x1cmp = 981.25; 
smax = 30; 
offlims=[0 inf] ; 

xshots = [];
for i=0:3
    subplot(2,2,i+1);
    [seismogram2,seis2,t]=shotrec(dx,dtstep,dt,tmax, ...
    vmodel,snap1,snap2,xrec,zrec,[5 10 30 40],0,1); 
    save('\Users\SHOTGATH\time.mat\','t');
    xshot= dx/2 + (dx * i) ;
    xshots = [xshots;xshot];
    [shot_nmo,xcmp,offsets,fold]= nmor_cmp(seismogram2,t,xrec,xshot,vrmsmod,xv,tv,dxcmp,x0cmp,x1cmp,1,smax,offlims);
    Path = '\Users\SHOTGATH\maped_to_cmp_gathers\';
    baseName = sprintf('ShotidNMO%d.mat', i+1);
    FullMatName = fullfile(Path,baseName); 
    save(FullMatName,'shot_nmo');
    imagesc(shot_nmo);colormap(gray);
    x_axis = [0 xaxes];z_axis=[0 zaxes];
    imagesc(x_axis,t_axis,shot_nmo);xlabel('cmp cor (m)','FontSize',7,'Color','k');ylabel('Time (s)','FontSize',7,'Color','k');
    set(gca,'Color','w','FontSize',8);colorbar;title('Amplitude');title(sprintf('SHOTID = %d',i+1),'FontSize',10,'Color','k');
end
save('\Users\SHOTGATH\xshots.mat\','xshots');



%% STACKING (Horizontally)
No_Shots = 80; 
[qq,pp] = size(shot_nmo);
load('\time.mat\','t');
load('\xrec.mat\','xrec');
load('\xshots.mat\','xshots');
load('\rmsvel_inTime.mat\','V_dix');
load('\Image_ray_input.mat\','Vs','I','x_IR','z_IR','t_IR');
Cr = movmean(V_dix,10,2); 
velrms = movmean(Cr,10,1);
cmp = [5, x0cmp, x1cmp]; 
xv = x_IR';
tv = t_IR';
offlims=[0 inf];
shots = zeros(qq,pp*No_Shots);
for i=1:No_Shots
    Path = '\\\maped_to_cmp_gathers\';
    baseName = sprintf('ShotidNMO%d.mat', i);
    FullMatName = fullfile(Path,baseName); 
    load(FullMatName);
    [stack,xcmp,stackfold,gathers,xoffs]=cmpstack(shots,t,xrec,xshots,cmp,velrms,tv,xv,2,2,10,offlims);
end



%%
function [seisw,seis,t]=shotrec(dx,dtstep,dt,tmax, ...
         model,snap1,snap2,xrec,zrec,wlet,tw,laplacian)



tic;
if(iscell(model))
    if(length(model)~=2)
        error('model must be a cell array of length 2 for density option')
    end
    velocity=model{1};
    density=log(model{2});
    denflag=1;
    if(size(velocity)~=size(density))
        error('Velocity model and density model must be matrices of the same size')
    end
else
    velocity=model;
    density=0;
    denflag=0;
end
boundary=2;

[nz,nx]=size(snap1);
if(prod(double(size(snap1)~=size(snap2))))
	error('snap1 and snap2 must be the same size');
end
if(prod(double(size(snap1)~=size(velocity))))
	error('snap1 and velocity must be the same size');
end

xmax=(nx-1)*dx;
zmax=(nz-1)*dx;


nrec=length(xrec);
if(nrec~=length(zrec))
	error('xrec and zrec are inconsistent')
end

test=between(0,xmax,xrec,2);
if(length(test)~=length(xrec))
	error('xrec not within grid')
end

test=between(0,zmax,zrec,2);
if(length(test)~=length(zrec))
	error('zrec not within grid')
end

ind=find(isnan(velocity), 1);
if(~isempty(ind))
   error('velocity model contains nans'); 
end

ind=find(isinf(velocity), 1);
if(~isempty(ind))
   error('velocity model contains infs'); 
end

vmin=min(min(velocity));
if(vmin<=0)
    error('zero or negative values found in velocity model');
end

if laplacian==1 
    if max(max(velocity))*dtstep/dx > 1/sqrt(2)
    	error('Model is unstable:  max(velocity)*dtstep/dx MUST BE < 1/sqrt(2)');
    end
elseif laplacian==2
    if max(max(velocity))*dtstep/dx > sqrt(3/8)
    	error('Model is unstable:  max(velocity)*dtstep/dx MUST BE < sqrt(3/8)');
    end
else
   error('invalid Laplacian flag')
end


if(abs(dt)<dtstep)
	error('abs(dt) cannot be less than dtstep')
end

if(length(wlet)==4)
   wlet=[wlet(2) wlet(2)-wlet(1) wlet(3) wlet(4)-wlet(3)];
   if(tw~=0 && tw ~=1)
      error('invalid phase flag');
   end
else
   if(length(wlet)~=length(tw))
      error('invalid wavelet specification')
   end
   w=wlet;
   wlet=0;
   if abs( tw(2)-tw(1)-abs(dt)) >= 0.000000001
      error('the temporal sampling rate of the wavelet and dt MUST be the same');
   end
end


seis=zeros(floor(tmax/dtstep),nrec);


ixrec = floor(xrec./dx)+1;
izrec = floor(zrec./dx)+1;


irec=(ixrec-1)*nz + izrec;



seis(1,:)=snap2(irec);

maxstep=round(tmax/dtstep)-1;
disp(['There are ' int2str(maxstep) ' steps to complete']);
time0=clock;


nwrite=2*round(maxstep/50)+1;
for k=1:2:maxstep
	
	
    if(denflag)
        snap1=afd_snap_acoustic2(dx,dtstep,velocity,density,snap1,snap2,laplacian,boundary);
        seis(k+1,:)=snap1(irec);
        snap2=afd_snap_acoustic2(dx,dtstep,velocity,density,snap2,snap1,laplacian,boundary);
        seis(k+2,:)=snap2(irec);
    else
        snap1=afd_snap(dx,dtstep,velocity,snap1,snap2,laplacian,boundary);
        seis(k+1,:)=snap1(irec);
        snap2=afd_snap(dx,dtstep,velocity,snap2,snap1,laplacian,boundary);
        seis(k+2,:)=snap2(irec);
    end
    
    
    if rem(k,nwrite) == 0
        timenow=clock;
        tottime=etime(timenow,time0);
        timeperstep=tottime/k;
        timeleft=timeperstep*(maxstep-k);
        
        disp(['wavefield propagated to ' num2str(k*dtstep) ...
            ' s; computation time remaining ' ...
            num2str(timeleft) ' s']);
    end

end


t=((0:size(seis,1)-1)*dtstep)';

disp('modelling completed')


if(abs(dt)~=dtstep)
	disp('resampling')
	phs=(sign(dt)+1)/2;
	dt=abs(dt);
	for k=1:nrec
		cs=polyfit(t,seis(:,k),4);
		[tmp,t2]=resamp(seis(:,k)-polyval(cs,t),t,dt,[min(t) max(t)],phs);
		seis(1:length(tmp),k)=tmp+polyval(cs,t2);
	end
	seis(length(t2)+1:length(t),:)=[];
	t=t2;
end
if(iscomplex(seis))
        yyy=1;
end


seisw=zeros(size(seis));
if(~wlet)
   nzero=near(tw,0);
   disp('applying wavelet');
	ifit=near(t,.9*max(t),max(t));
	tpad=(max(t):dt:1.1*max(t))';
   for k=1:nrec
		tmp=seis(:,k);
		cs=polyfit(t(ifit),tmp(ifit),1);
		tmp=[tmp;polyval(cs,tpad)];
      tmp2=convz(tmp,w,nzero);
		seisw(:,k)=tmp2(1:length(t));
   end
else
   disp('filtering...')
	ifit=near(t,.9*max(t),max(t));

	tpad=(max(t)+dt:dt:1.1*max(t))';
   for k=1:nrec
		tmp=seis(:,k);
		cs=polyfit(t(ifit),tmp(ifit),1);
		tmp=[tmp;polyval(cs,tpad)];

        tmp2=filtf(tmp,[t;tpad],[wlet(1) wlet(2)],[wlet(3) wlet(4)],tw);
		seisw(:,k)=tmp2(1:length(t));
   end
end

if(iscomplex(seisw))

        seisw=real(seisw);
end

toc;







