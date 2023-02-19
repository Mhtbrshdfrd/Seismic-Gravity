function Model_f = tomofast_grid(dens_const,nx,ny,nz,x_min,x_max,y_min,y_max,z_min,z_max,d_size_mesh)


XX = linspace(x_min,x_max,nx);
YY = linspace(y_min,y_max,ny);
ZZ = linspace(z_min,z_max,nz);


xx = XX'; yy = YY; zz = ZZ;
nx_c = nx; ny_c = ny; nz_c = nz;
n_for_x = ny*nz; n_for_y = nx*nz; n_for_z = ny*nx;



X = repmat(xx,n_for_x,1);
INDX = repmat(1:length(xx),1,n_for_x);
% y
yy = yy';
Y = [];
IND_Y = [];
for i=1:ny_c
    Y = [Y;repmat(yy(i),nx_c,1)];
    IND_Y=[IND_Y;repmat(i,nx_c,1)];
end
Y = repmat(Y,nz_c,1);
INDY=repmat(IND_Y,nz_c,1);
% z
zz =zz';
Z = [];
INDZ = [];
for j=1:nz_c
    Z = [Z;repmat(zz(j),n_for_z,1)];
    INDZ=[INDZ;repmat(j,n_for_z,1)];
end
% X1 & X2
X1 = X ;
X2 = X + (d_size_mesh);
Y1 = Y ;
Y2 = Y + (d_size_mesh);
Z1 = Z ;
Z2 = Z + (d_size_mesh);
Model_f = ones(length(X),11);
Model_f(:,1) = X1;
Model_f(:,2) = X2;
Model_f(:,3) = Y1;
Model_f(:,4) = Y2;
Model_f(:,5) = Z1;
Model_f(:,6) = Z2;
Model_f(:,8) = INDX;
Model_f(:,9) = INDY;
Model_f(:,10) = INDZ;
Model_f(:,7) = dens_const;
end
