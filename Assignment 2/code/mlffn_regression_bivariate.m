trainData =load('data/bivariate_data/group12_train.txt');
% train_X=normr(trainData(:,1:2)');
train_X=trainData(:,1:2)';

train_t=trainData(:,3)';


valData =load('data/bivariate_data/group12_val.txt');
% val_X=normr(valData(:,1:2)');
val_X=valData(:,1:2)';
val_t=valData(:,3)';

testData =load('data/bivariate_data/group12_test.txt');
% test_X=normr(testData(:,1:2)');
test_X=testData(:,1:2)';
test_t=testData(:,3)'; 
% data =[train_X,val_X,test_X];

 X= [3:15];
%for k =1:20
     Y_train=zeros(size(X));
     Y_val=zeros(size(X));
     Y_test=zeros(size(X));
min_mse =10000;
min_mse_h1=10000;

for no_hiddennodes =3:10
    minmax_train =minmax(train_X);
no_outputnodes =1;
h2_nodes=0;
lr =0.1 
mnet =newff(train_X,train_t,[no_hiddennodes],{'tansig','purelin'},'trainlm','learngd','mse');
mnet.trainParam.lr =lr;
%mnet.trainParam.mc=0.8;
mnet.trainParam.epochs = 1000;
[net] =train(mnet,train_X,train_t,[],[]);

[train_predict] =sim(net,train_X);
train_mse = (sum(sum((train_t-train_predict).*(train_t-train_predict))))/size(train_X,2);

[val_predict] =sim(net,val_X);
val_mse = (sum(sum((val_t-val_predict).*(val_t-val_predict))))/size(val_X,2);

[test_predict] =sim(net,test_X);
test_mse = (sum(sum((test_t-test_predict).*(test_t-test_predict))))/size(test_X,2);


if val_mse <min_mse_h1
    min_mse_h1 =val_mse;
    min_valP_h1 =val_predict;
    min_hiddennodes_h1= no_hiddennodes;
    min_net_h1 =net;
    min_lr_h1 =lr;
end;
Y_train(1,no_hiddennodes-2)=train_mse;
Y_val(1,no_hiddennodes-2)=val_mse;
Y_test(1,no_hiddennodes-2)=test_mse;


h2_upper =no_hiddennodes-1;
for h2_nodes =2:h2_upper
minmax_train =minmax(train_X);
no_outputnodes =1;
% for lr= 0.1:0.1:0.9
mnet =newff(train_X,train_t,[no_hiddennodes,h2_nodes],{'tansig','tansig','purelin'},'traingd','learngd','mse');
mnet.trainParam.lr = lr;
%mnet.trainParam.mc=0.8;
mnet.trainParam.epochs = 1000;
[net] =train(mnet,train_X,train_t,[],[]);
[val_predict] =sim(net,val_X);
val_mse = (sum(sum((val_t-val_predict).*(val_t-val_predict))))/size(val_X,2);


if val_mse <min_mse
    min_mse =val_mse;
    min_valP =val_predict;
    min_hiddennodes= no_hiddennodes;
    min_h2_nodes= h2_nodes;
    min_net =net;
    min_lr =lr

 end;


% end;
%Y(1,no_hiddennodes-2)=mse;
  end;
 
end;



% end;
%% One hidden layer
[train_predict] =sim(min_net_h1,train_X);
 train_mse = (sum(sum((train_t-train_predict).*(train_t-train_predict))))/size(train_X,2);
disp(sprintf('train MSE = %f',train_mse));
fig=figure();
plot(train_t,train_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/exp/modeltarget_hl' num2str(no_hiddennodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

figure();
plot3(train_X(1,:),train_X(2,:),train_t,'g*');
hold on;
plot3(train_X(1,:),train_X(2,:),train_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/exp/scatterplot_train_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');


[val_predict] =sim(min_net_h1,val_X);
val_mse = (sum(sum((val_t-val_predict).*(val_t-val_predict))))/size(val_X,2);
disp(sprintf('val MSE = %f',val_mse));
fig=figure();
plot(val_t,val_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/exp/modeltarget_val_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

fig =figure();
plot3(val_X(1,:),val_X(2,:),val_t,'g*');
hold on;
plot3(val_X(1,:),val_X(2,:),val_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/scatterplot_valn_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');


[test_predict] =sim(min_net_h1,test_X);
test_mse = (sum(sum((test_t-test_predict).*(test_t-test_predict))))/size(test_X,2);
disp(sprintf('test MSE = %f',test_mse));
fig=figure();
plot(test_t,test_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/modeltarget_test_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

fig=figure();
plot3(test_X(1,:),test_X(2,:),test_t,'g*');
hold on;
plot3(test_X(1,:),test_X(2,:),test_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/scatterplot_test_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');


beta=0.8
h1_weights= min_net_h1.IW{1,1};
h1_bias =min_net_h1.b{1,1}

w0 = [];
for i =1:min_hiddennodes_h1

w0  =[w0;(ones(1,2000)*h1_bias(i))];

end;
h1_a_c1 =(train_X'*h1_weights')'+w0;
n_c1 =beta*h1_a_c1;
n1_nodes_values_c1= tansig(n_c1);
for ni =1:min_hiddennodes_h1
fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(ni,:)),'r*');
%  xlim([-5,5]);
%  ylim([-5,5]);
% zlim([0,0.5])
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
ttl =['Hidden layer 1 - Hidden node', ni];
title(ttl);
%fn=['plots/bivariate/scurve_betah1_hl1_hn' ni]
%print(fig,fn,'-dpng');
end;

[out] =sim(min_net_h1,train_X);
fig=figure();
plot3(train_X(2,:)',train_X(1,:)',out,'r*');
%  xlim([-5,5]);
%  ylim([-5,5]);
% zlim([0,0.5])
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
fn=['plots/bivariate/scurve_betah1_output']
print(fig,fn,'-dpng');

%Two hidden layer
[train_predict] =sim(min_net,train_X);
 train_mse = (sum(sum((train_t-train_predict).*(train_t-train_predict))))/size(train_X,2);
disp(sprintf('train MSE = %f',train_mse));
fig=figure();
plot(train_t,train_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/modeltarget_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

figure();
plot3(train_X(1,:),train_X(2,:),train_t,'g*');
hold on;
plot3(train_X(1,:),train_X(2,:),train_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/scatterplot_train_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');


[val_predict] =sim(min_net,val_X);
val_mse = (sum(sum((val_t-val_predict).*(val_t-val_predict))))/size(val_X,2);
disp(sprintf('val MSE = %f',val_mse));
fig=figure();
plot(val_t,val_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/modeltarget_val_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

fig =figure();
plot3(val_X(1,:),val_X(2,:),val_t,'g*');
hold on;
plot3(val_X(1,:),val_X(2,:),val_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/scatterplot_valn_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

[test_predict] =sim(min_net,test_X);
test_mse = (sum(sum((test_t-test_predict).*(test_t-test_predict))))/size(test_X,2);
disp(sprintf('test MSE = %f',test_mse));
fig=figure();
plot(test_t,test_predict,'*');
xlabel('target');
ylabel('model');
fn=['plots/bivariate/modeltarget_test_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');

fig=figure();
plot3(test_X(1,:),test_X(2,:),test_t,'g*');
hold on;
plot3(test_X(1,:),test_X(2,:),test_predict,'r*');
xlabel('Input');
ylabel('Output');
legend('Target','Model');
fn=['plots/bivariate/scatterplot_test_hl' num2str(no_hiddennodes) '_hl2' num2str(h2_nodes) '_lr' num2str(lr*10)]
print(fig,fn,'-dpng');


beta=0.8
h1_weights= min_net.IW{1,1};
h1_bias =min_net.b{1,1}
w0 = [];
for i =1:min_hiddennodes

w0  =[w0;(ones(1,2000)*h1_bias(i))];

end;
h1_a_c1 =(train_X'*h1_weights')'+w0;
n_c1 =beta*h1_a_c1;
n1_nodes_values_c1= tansig(n_c1);
for nj =1:min_hiddennodes
fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(nj,:)),'r*');
%  xlim([-5,5]);
%  ylim([-5,5]);
% zlim([0,0.5])
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
ttl=['Hidden Layer 1 - Hidden node ' nj];
title(ttl);
% fn=['plots/bivariate/scurve_betah2_hl1']
% print(fig,fn,'-dpng');
end;
h2_weights= min_net.LW{2,1};
h2_bias =min_net.b{2,1}
w0 = [];
for i =1:min_h2_nodes

w0  =[w0;(ones(1,2000)*h2_bias(i))];

end;
h2_a_c1 =(n1_nodes_values_c1'*h2_weights')'+w0;
n_c1 =beta*h2_a_c1;
n2_nodes_values_c1= tansig(n_c1);
for ni=1:min_h2_nodes
fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(ni,:)),'r*');
%  xlim([-5,5]);
%  ylim([-5,5]);
% zlim([0,0.5])
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
ttl=['Hidden Layer 2 - Hidden node ' ni];
title(ttl);
% fn=['plots/bivariate/scurve_betah2_hl2']
% print(fig,fn,'-dpng');
end;
% out_weights= min_net.LW{3,2};
% h2_bias =min_net.b{3,1}
% w0=(ones(1,2000)*h2_bias);
% 
% out_a =(n2_nodes_values_c1'*out_weights')'+w0;
% n =beta*out_a;
% out_nodes= tansig(n);
% fig= figure();
% plot3(train_X(2,:)',train_X(1,:)',tansig(n(1,:)),'r*');
[out] =sim(min_net,train_X);
fig=figure();
plot3(train_X(2,:)',train_X(1,:)',out,'r*');
%  xlim([-5,5]);
%  ylim([-5,5]);
% zlim([0,0.5])
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
fn=['plots/bivariate/scurve_betah2_output']
print(fig,fn,'-dpng');
% perf = perform(net,predict,val_t);
 %confusionmat(val_t,predict)
 %disp(Evaluate(val_t,predict));

%  plotconfusion(val_t,min_valP);
% 

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
 
 disp(sprintf('Min no of hidden nodes =%d h2 = %d',min_hiddennodes,min_h2_nodes));



%%