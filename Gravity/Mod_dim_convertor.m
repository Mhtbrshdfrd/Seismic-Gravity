function [Model_2d_f,Model_2d,nlset,X_ind,Y_ind,Z_ind] = Mod_dim_convertor(d_size_mesh,nx,ny,nz, x_min, x_max,y_min,y_max,z_min,z_max,model_3d,dens_const_prior10,s,cdp_x,cdp_y)
XX = linspace(x_min,x_max,nx);
YY = linspace(y_min,y_max,ny);
ZZ = linspace(z_min,z_max,nz);
X1 = model_3d(:,1);
X11 = round(X1, 4);
X2 = model_3d(:,2);
X = double((X1 + X2)/2);
Y1 = model_3d(:,3);
Y11 = round(Y1, 4);
Y2 = model_3d(:,4);
Y = double((Y1 + Y2)/2);
Z1 = model_3d(:,5);
Z2 = model_3d(:,6);
Z = double((Z1 + Z2)/2);
model_3d(:,7) = dens_const_prior10;
rho_val = dens_const_prior10;
nlset = length(unique(rho_val));
ending = floor(length(cdp_x)/s)*s;
indices = [];
for i=1:s:ending-s
    x1 = cdp_x(i);
    x2 = cdp_x(i+s);
    y1 = cdp_y(i);
    y2 = cdp_y(i+s);
    [VALx1, INDx1] = min(abs(XX - x1));
    [VALx2, INDx2] = min(abs(XX - x2));
    [VALy1, INDy1] = min(abs(YY - y1));
    [VALy2, INDy2] = min(abs(YY - y2));
    [Xbr,Ybr]=bresenham(INDx1,INDy1,INDx2,INDy2); %indices of wanted cells
     xXx = XX(Xbr);
     yYy = YY(Ybr);
     lenx = length(Xbr);
     for j=1:lenx
         indices = [indices;find(X1 == xXx(j) & Y1 == yYy(j))];
     end
end
Model_2d = model_3d(indices,:);
Model_2d_f = Model_2d;
X_index = Model_2d_f(:,8);
Y_index = Model_2d_f(:,9);
Len_line = length(indices)/nz;
del_rows = [];
for k= 1:Len_line-1
    if X_index(k * nz)== X_index(k * nz+1) 
        del_rows = [del_rows;(k*nz)+1];
    end
end
for h=1:length(del_rows)
    Model_2d_f(del_rows(h):del_rows(h)+nz-1,:)=0;
end
%delete zero rows
Model_2d_f( ~any(Model_2d_f,2), : ) = [];
X_ind = Model_2d_f(:,8);
Y_ind = Model_2d_f(:,9);
Z_ind = Model_2d_f(:,10);