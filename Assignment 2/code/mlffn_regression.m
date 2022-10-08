trainData =load('data/univariate/data_50.txt');
%train_X=normr(trainData(:,1:2)');
train_X=trainData(:,1)';

train_t=trainData(:,2)';


valData =load('data/univariate/data_30.txt');
% val_X=normr(valData(:,1:2)');
val_X=valData(:,1)';
val_t=valData(:,2)';

testData =load('data/univariate/data_100.txt');
% test_X=normr(testData(:,1:2)');
test_X=testData(:,1)';
test_t=testData(:,2)'; 

X= [2:15];
%for k =1:20
    Y_train=zeros(size(X));
    Y_val=zeros(size(X));
    Y_test=zeros(size(X));
min_mse =10000;
for no_hiddennodes =2:15
  
minmax_train =minmax(train_X);
no_outputnodes =1;
mnet =newff(train_X,train_t,[no_hiddennodes],{'tansig','purelin'},'traingd','learngd','mse');
mnet.trainParam.lr = 0.3;
%mnet.trainParam.mc=0.8;
mnet.trainParam.epochs = 500;
[net] =train(mnet,train_X,train_t,[],[]);
[predict] =sim(net,train_X);
train_mse = (sum(sum((train_t-predict).*(train_t-predict))));

[predict] =sim(net,val_X);
val_mse = (sum(sum((val_t-predict).*(val_t-predict))));

[predict] =sim(net,test_X);
test_mse = (sum(sum((test_t-predict).*(test_t-predict))));
if val_mse <min_mse
    min_mse =val_mse;
    min_valP =predict;
    min_hiddennodes= no_hiddennodes;
    min_net =net;
end;
Y_train(1,no_hiddennodes-1)=train_mse;
Y_val(1,no_hiddennodes-1)=val_mse;
Y_test(1,no_hiddennodes-1)=test_mse;

end;

%  perf = perform(net,predict,val_t);
 %confusionmat(val_t,predict)
 %disp(Evaluate(val_t,predict));

%  plotconfusion(val_t,min_valP);
% 
[predict] =sim(min_net,train_X);
mse = (sum(sum((train_t-predict).*(train_t-predict))))/size(train_X,2);
disp(sprintf('train MSE = %f',mse));
 figure();
plot(train_t,predict,'*');
xlabel('target');
ylabel('model');

figure();
plot(train_X,train_t,'g*');
hold on;
plot(train_X,predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
[predict] =sim(min_net,val_X);
mse = (sum(sum((val_t-predict).*(val_t-predict))))/size(val_X,2);
disp(sprintf('val MSE = %f',mse));
figure();
plot(val_t,predict,'*');
xlabel('target');
ylabel('model');
figure();
plot(val_X,val_t,'g*');
hold on;
plot(val_X,predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
[predict] =sim(min_net,test_X);
mse = (sum(sum((test_t-predict).*(test_t-predict))))/size(test_X,2);
disp(sprintf('test MSE = %f',mse));
figure();
plot(test_t,predict,'*');
xlabel('target');
ylabel('model');
figure();
plot(test_X,test_t,'g*');
hold on;
plot(test_X,predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');

figure();
plot(X,Y_train,'b','LineWidth',2);
hold on;
plot(X,Y_val,'g','LineWidth',2);
hold on;
plot(X,Y_test,'r','LineWidth',2);
xlabel('No of Hidden Nodes');
ylabel('MSE');
legend('Train','Validation','Test')
title('MSE vs No of hidden nodes');

% figure();
% plot(X,Y_val,'LineWidth',2);
% xlabel('No of Hidden Nodes');
% ylabel('MSE');
% title('MSE vs No of hidden nodes for Validation data');
% 
% figure();
% plot(X,Y_test,'LineWidth',2);
% xlabel('No of Hidden Nodes');
% ylabel('MSE');
% title('MSE vs No of hidden nodes for Testing data');
 
 disp(sprintf('Min no of hidden nodes =%d',min_hiddennodes));




%%