
x1 = linspace(0, 0.5, 100)';
y1=cos(2*pi*x1).*cos(2*pi*x1);
figure;
plot(x1,y1,'b-','LineWidth',2);
xlabel('x')'
ylabel('f(x)=cos^2(2*pi*x)');
title('cos^2(2*pi*x)');
print('plots/underlying_fn.jpg');

data =dlmread('data/data_10.txt',' ');
x= data(:,1);
t=data(:,2);
p= polycurvefit_reg(x,t,9,0)
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
xi = linspace(0, 0.5, 100);
y_predict = poly_out_values(p, x);
yi= interp1 (x, y_predict, xi,'spline');
plot(x, t, "*",  xi, yi, "r-",'LineWidth',2);
#plot(x, t, "*", x, poly_out_values(p, x), "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('ln lambda= - infinity');
print('plots/reg_poly_curve_N_10_M_9_lamda-inf.jpg')

p=polycurvefit_reg(x,t,9,e^(-18));
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
xi = linspace(0, 0.5, 100);
y_predict = poly_out_values(p, x);
yi= interp1 (x, y_predict, xi,'spline');
plot(x, t, "*",  xi, yi, "r-",'LineWidth',2);
#plot(x, t, "*", x, poly_out_values(p, x), "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('ln lamda = -18');
print('plots/reg_poly_curve_N_10_M_9_lamda-18.jpg')

p=polycurvefit_reg(x,t,9,1);
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
xi = linspace(0, 0.5, 100);
y_predict = poly_out_values(p, x);
yi= interp1 (x, y_predict, xi,'spline');
plot(x, t, "*",  xi, yi, "r-",'LineWidth',2);
#plot(x, t, "*", x, poly_out_values(p, x), "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('ln lamda = 0');
print('plots/reg_poly_curve_N_10_M_9_lamda-0.jpg')
