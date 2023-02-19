%% %% Simple code for straight ray-tracing of seismic travel times


%% 


load('C:\Users\Data\syntheticData\VelMod_original.mat','V');% Load Velocity Model
Model = V;
SOURCE = [10 3]; 
RECEIVER = [600 10]; 
dx=10; 
dz=5;  
dray=0.005;
dr=22; 
%% 


[iModelX,iModelZ,iModelS,xmin,xmax,zmin,zmax,F] = ReadModels(Model,dx,dz);

%% refract rays

[ray0,T0] = Refracting(SOURCE,RECEIVER,dz,dray,dr,F); 



%% visual
h=pcolor(iModelX',iModelZ',Model);
set(gca,'Ydir','reverse')
set(h, 'EdgeColor', 'none');
colormap(flipud(jet))
colorbar
xlim([xmin xmax])
ylim([zmin zmax])
hold on
plot(SOURCE(1),SOURCE(2),'h','MarkerFaceColor','r','MarkerEdgeColor','r','MarkerSize',7.5)
plot(RECEIVER(1),RECEIVER(2),'v','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',7.5)
plot(ray0(:,1),ray0(:,2),'-k','LineWidth', 0.5)



%% FIX INPUT DATA

function [iModelX,iModelZ,iModelS,xmin,xmax,zmin,zmax,F] = ReadModels(Model,dx,dz)

ModelS=1./Model;
 
vx=0:dx:(size(Model,2)-1)*dx; 
vz=(0:dz:(size(Model,1)-1)*dz)'; 
ModelX=repmat(vx,size(Model,1),1); 
ModelZ=repmat(vz,1,size(Model,2)); 

iModelS=ModelS';
iModelX=ModelX';
iModelZ=ModelZ';
zmax=max(vz); 
zmin=min(vz); 
xmax=max(vx); 
xmin=min(vx); 
F=griddedInterpolant(iModelX,iModelZ,iModelS); 
end

%%
function [ray0,T0] = Refracting(SOURCE,RECEIVER,dz,dray,dr,F)


ray0=[SOURCE(1):(RECEIVER(1)-SOURCE(1))/dr:RECEIVER(1);SOURCE(2):(RECEIVER(2)-SOURCE(2))/dr:RECEIVER(2)]';

T0=GetTime(ray0,F); 
movedz=dz*dray;
NCC=0;
while NCC<dr-1
    NCC=0;
    for j=2:dr

        rayz1(:,1)=[ray0(1:j-1,2); ray0(j,2)+movedz; ray0(j+1:end,2)];

 
        T1=GetTime([ray0(:,1) rayz1],F);
        rayz2(:,1)=[ray0(1:j-1,2); ray0(j,2)-movedz; ray0(j+1:end,2)];


        T2=GetTime([ray0(:,1) rayz2],F);
        
        if T1<T0 && T1<T2 
            ray0=[ray0(:,1) rayz1];
            T0=T1;
        elseif T2<T0 && T2<T1 
            ray0=[ray0(:,1) rayz1];
            T0=T2;
        else
            NCC=NCC+1; 
        end
       
    end
end
end

%% 
function [T0] = GetTime(ray0,F)
Snodes = F(ray0(:,1), ray0(:,2)); 
B=mean([Snodes(1:end-1)';Snodes(2:end)']); 
C=sqrt((diff(ray0(:,1)).*diff(ray0(:,1)))+(diff(ray0(:,2)).*diff(ray0(:,2)))); 
T0=B*C; 
end

