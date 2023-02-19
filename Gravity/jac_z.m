function Jac_Z = jac_z(Xdata,Ydata,Zdata,X,Y,Z,nx,ny,nz,d_size_mesh)

tic
G=6.67e-08;
num_cells=nx*ny*nz; 
n_data=numel(Xdata); 

nel_tot=num_cells*n_data;  
fprintf('Total of: %d elements. \n',nel_tot)

grid=zeros(nx*ny*nz,6); 

ind=1; 
for k=1:nz
    for j=1:ny
        for i=1:nx
            grid(ind,1)=X(ind)-d_size_mesh/2;
            grid(ind,2)=X(ind)+d_size_mesh/2;
            grid(ind,3)=Y(ind)-d_size_mesh/2;
            grid(ind,4)=Y(ind)+d_size_mesh/2;
            grid(ind,5)=Z(ind)-d_size_mesh/2;
            grid(ind,6)=Z(ind)+d_size_mesh/2;
            ind=ind+1; 
        end
    end
end

JacZ=zeros(n_data,num_cells); 

XX = zeros(2,1);
YY = zeros(2,1);
ZZ = zeros(2,1);

signo = [-1 ,1];

LineZ = zeros(1,num_cells);
Zdata=unique(Zdata); 

for i=1:n_data
    
    for j=1:num_cells
        
        XX(1) = Xdata(i) - grid(j,1);
        XX(2) = Xdata(i) - grid(j,2);
        YY(1) = Ydata(i) - grid(j,3);
        YY(2) = Ydata(i) - grid(j,4);
        ZZ(1) = Zdata - grid(j,5);
        ZZ(2) = Zdata - grid(j,6);
        
        gz = 0;
        
        for K = 1:2
            for L = 1:2
                for M = 1:2
                    dmu = signo(K) * signo(L) * signo(M);
                    
                    Rs = sqrt(XX(K).^2 + YY(L).^2 + ZZ(M).^2);

                    arg3 = atan2(XX(K)*YY(L), ZZ(M)*Rs);

                    if (arg3 < 0)
                        arg3 = arg3 + 6.283185306;
                    end
                    
                    arg4 = Rs + XX(K);
                    arg5 = Rs + YY(L);

                    arg4 = log(arg4);
                    arg5 = log(arg5);
 
                    gz = gz + dmu * (ZZ(M)*arg3 - XX(K)*arg5 - YY(L)*arg4);
                end
            end
        end
        
        
        LineZ(j) = gz;
    end
    JacZ(i,:)=LineZ;
    fprintf('Done: %d perc \n',round(100*(i-1)/n_data))
end

Jac_Z=JacZ*G; 

toc
end