function [new_data,Xdata,Ydata,Zdata,data,n_data] = Grav_data_cut(Grav_data_and_coor,pad, d_size_mesh,step,x_min,x_max,y_min,y_max)
limit = pad*d_size_mesh;
lx_l = x_min + limit;
ux_l = x_max - limit;
ly_l = y_min + limit;
uy_l = y_max - limit;
Xdata = Grav_data_and_coor(:,1);
Ydata = Grav_data_and_coor(:,2);
data =  (Grav_data_and_coor(:,4))*1e-5;

x_r = find(Xdata > lx_l & Xdata < ux_l);  
y_r = find(Ydata > ly_l & Ydata < uy_l);
rr = intersect(x_r,y_r);
%reduce the data space size?
RR = rr(1:step:end);
KK = rr(1:step:end);
hh = unique(reshape([RR;KK],1,[]),'stable');
Xdata = Xdata(hh);
Ydata = Ydata(hh); 
Zdata = -0.1 * ones(size(Xdata));
data = data(hh);
n_data = length(data);

new_data = [Xdata Ydata Zdata data];
end 