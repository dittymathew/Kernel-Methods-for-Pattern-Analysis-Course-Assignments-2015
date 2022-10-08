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
data =[train_X';val_X';test_X'];
norm_data=zscore(data);
train_X = norm_data(1:2000,:)';
val_X=norm_data(2001:2300,:)';
test_X =norm_data(2301:2500,:)';

 X= [3:15];
%for k =1:20
     Y_train=zeros(size(X));
     Y_val=zeros(size(X));
     Y_test=zeros(size(X));
min_mse =10000;
min_mse_h1=10000;

% for no_hiddennodes =3:15
    minmax_train =minmax(train_X);
no_outputnodes =1;
h2_nodes=0;
lr =0.1 
mnet =newff(train_X,train_t,[3,2],{'tansig','tansig','purelin'},'trainlm','learngd','mse');
mnet.trainParam.lr =lr;
%mnet.trainParam.mc=0.8;
mnet.trainParam.epochs = 1;
[net] =train(mnet,train_X,train_t,[],[]);

% end;
%% One hidden layer

% beta=0.8
% h1_weights= net.IW{1,1};
% h1_bias =net.b{1,1}
% 
% w0 = [];
% for i =1:3
% 
% w0  =[w0;(ones(1,2000)*h1_bias(i))];
% 
% end;
% h1_a_c1 =(train_X'*h1_weights')'+w0;
% n_c1 =beta*h1_a_c1;
% n1_nodes_values_c1= tansig(n_c1);
% 
% fig= figure();
% plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(1,:)),'r*');
% set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
% xlabel('feature 1');
% ylabel('feature 2');
% zlabel('Hidden layer');
% title('Hidden layer 1 - Hidden node 1');
% 
% fig= figure();
% plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(2,:)),'r*');
% set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
% xlabel('feature 1');
% ylabel('feature 2');
% zlabel('Hidden layer');
% title('Hidden layer 1 - Hidden node 2');
% 
% fig= figure();
% plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(3,:)),'r*');
% set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
% xlabel('feature 1');
% ylabel('feature 2');
% zlabel('Hidden layer');
% title('Hidden layer 1 - Hidden node 3');
% 
% 
% %fn=['plots/bivariate/scurve_betah1_hl1_hn' ni]
% %print(fig,fn,'-dpng');
% 
% 
% [out] =sim(net,train_X);
% fig=figure();
% plot3(train_X(2,:)',train_X(1,:)',out,'r*');
% %  xlim([-5,5]);
% %  ylim([-5,5]);
% % zlim([0,0.5])
% set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
% xlabel('feature 1');
% ylabel('feature 2');
% zlabel('Output layer');
% fn=['plots/bivariate/scurve_betah1_output']
% print(fig,fn,'-dpng');

%Two hidden layer

beta=0.8
h1_weights= net.IW{1,1};
h1_bias =net.b{1,1}
w0 = [];
for i =1:3

w0  =[w0;(ones(1,2000)*h1_bias(i))];

end;
h1_a_c1 =(train_X'*h1_weights')'+w0;
n_c1 =beta*h1_a_c1;
n1_nodes_values_c1= tansig(n_c1);

fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(1,:)),'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
title('Hidden Layer 1 - Hidden node 1');

fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(2,:)),'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
title('Hidden Layer 1 - Hidden node 2');


fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(3,:)),'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
title('Hidden Layer 1 - Hidden node 3');

h2_weights= net.LW{2,1};
h2_bias =net.b{2,1}
w0 = [];
for i =1:2
w0  =[w0;(ones(1,2000)*h2_bias(i))];

end;
h2_a_c1 =(n1_nodes_values_c1'*h2_weights')'+w0;
n_c1 =beta*h2_a_c1;
n2_nodes_values_c1= tansig(n_c1);

fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(1,:)),'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
title('Hidden Layer 2 - Hidden node 1');

fig= figure();
plot3(train_X(2,:)',train_X(1,:)',tansig(n_c1(2,:)),'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer');
title('Hidden Layer 2 - Hidden node 2');

[out] =sim(net,train_X);
fig=figure();
plot3(train_X(2,:)',train_X(1,:)',out,'r*');
set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');


