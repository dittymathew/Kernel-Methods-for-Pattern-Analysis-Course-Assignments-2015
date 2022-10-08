
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
p= polycurvefit(x,t,0)
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
plot(x, t, "*", x, poly_out_values(p, x), "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('N=10, M=0');
print('plots/poly_curve_N_10_M_0.jpg')

p=polycurvefit(x,t,1);
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
plot(x, t, "*", x, poly_out_values(p, x), "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('N=10, M=1');
print('plots/poly_curve_N_10_M_1.jpg');

p=polycurvefit(x,t,3);
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
title('N=10, M=3');
print('plots/poly_curve_N_10_M_3.jpg');


p=polycurvefit(x,t,6);
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
title('N=10, M=6');
print('plots/poly_curve_N_10_M_6.jpg');

p=polycurvefit(x,t,9);
figure;
plot(x1,y1,'b-','LineWidth',2);
hold on;
xi = linspace(0, 0.5, 100);
y_predict = poly_out_values(p, x);
yi= interp1 (x, y_predict, xi,'spline');
plot(x, t, "*",  xi, yi, "r-",'LineWidth',2);
xlabel('x');
ylabel('t');
title('N=10, M=9');
print('plots/poly_curve_N_10_M_9.jpg');


data =dlmread('data/data_15.txt',' ');
x= data(:,1);
t=data(:,2);

p=polycurvefit(x,t,9);
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
title('N=15, M=9');
print('plots/poly_curve_N_15_M_9.jpg');

data =dlmread('data/data_100.txt',' ');
x= data(:,1);
t=data(:,2);

p=polycurvefit(x,t,9);
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
title('N=100, M=9');
print('plots/poly_curve_N_100_M_9.jpg');

