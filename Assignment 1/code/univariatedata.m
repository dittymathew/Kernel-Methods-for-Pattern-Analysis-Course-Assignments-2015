num_points=10
x = linspace(0, 0.5, num_points);
y=cos(2*pi*x).*cos(2*pi*x) +normrnd(0,0.09,1,num_points);
figure()
plot(x, y,'g');

data=[transpose(x),transpose(y)];
dlmwrite('data/validationdata_10.txt', data,' ');
