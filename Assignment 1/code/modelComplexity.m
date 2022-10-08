
data =dlmread('data/data_10.txt',' ');
train_x= data(:,1);
train_y=data(:,2);
data =dlmread('data/validationdata_10.txt',' ');
val_x= data(:,1);
val_y=data(:,2);
data =dlmread('data/testdata_100.txt',' ');
test_x= data(:,1);
test_y=data(:,2);
train_error=zeros(1,10);
test_error=zeros(1,10);
val_error=zeros(1,10);
for M= 0:9;
  w= polycurvefit(train_x,train_y,M);
  train_predict =poly_out_values(w,train_x);
  train_error(1,M+1) = rsserror(train_predict,train_y);
  #w= polycurvefit(val_x,val_y,M);
  val_predict =poly_out_values(w,val_x);
  val_error(1,M+1) = rsserror(val_predict,val_y);
  test_predict =poly_out_values(w,test_x);
  test_error(1,M+1) = rsserror(test_predict,test_y);
endfor;
figure;
plot(0:9,train_error,'r-*','LineWidth',2,0:9,val_error,'g-*','LineWidth',2,0:9,test_error,'b-*','LineWidth',2);
xlabel('M');
ylabel('RMS Error');
legend('Train','Validation','Test');
print('plots/rmserror.jpg')

train_error=zeros(1,40);
test_error=zeros(1,40);
val_error=zeros(1,40);
for i =-39:0;
  lamda = e^i
  w= polycurvefit_reg(train_x,train_y,9,lamda);
  train_predict =poly_out_values(w,train_x);
  train_error(1,i+40) = rsserror(train_predict,train_y);
  w= polycurvefit_reg(train_x,train_y,9,lamda);
  val_predict =poly_out_values(w,val_x);
  val_error(1,i+40) = rsserror(val_predict,val_y);
  test_predict =poly_out_values(w,test_x);
  test_error(1,i+40) = rsserror(test_predict,test_y);
endfor
figure;
plot(-39:0,train_error,'r-','LineWidth',2,-39:0,val_error,'g-','LineWidth',2,-39:0,test_error,'b-','LineWidth',2);
xlabel('ln \lambda');
ylabel('RMS Error');
legend('Train','Validation','Test');
print('plots/rmserror_reg.jpg')
  
