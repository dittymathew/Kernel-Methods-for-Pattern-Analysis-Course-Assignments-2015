train_cls1 =load('data/nonlinearlySeparable/group12/class1_train.txt');
train_cls2 =load('data/nonlinearlySeparable/group12/class2_train.txt');
y_train_cls1= [ones(size(train_cls1,1),1),zeros(size(train_cls1,1),1)];
y_train_cls2= [zeros(size(train_cls2,1),1),ones(size(train_cls2,1),1)];

train_cls1 =[train_cls1,y_train_cls1];
train_cls2 =[train_cls2,y_train_cls2];

trainData =[train_cls1;train_cls2];
%train_X=normr(trainData(:,1:2)');
train_X=trainData(:,1:2)';
train_t=trainData(:,3:4)';


val_cls1 =load('data/nonlinearlySeparable/group12/class1_val.txt');
val_cls2 =load('data/nonlinearlySeparable/group12/class2_val.txt');

y_val_cls1= [ones(size(val_cls1,1),1),zeros(size(val_cls1,1),1)];
y_val_cls2= [zeros(size(val_cls2,1),1),ones(size(val_cls2,1),1)];

val_cls1 =[val_cls1,y_val_cls1];
val_cls2 =[val_cls2,y_val_cls2];

valData =[val_cls1;val_cls2];
%val_X=normr(valData(:,1:2)');
val_X=valData(:,1:2)';
val_t=valData(:,3:4)';

test_cls1 =load('data/nonlinearlySeparable/group12/class1_test.txt');
test_cls2 =load('data/nonlinearlySeparable/group12/class2_test.txt');


y_test_cls1= [ones(size(test_cls1,1),1),zeros(size(test_cls1,1),1)];

y_test_cls2= [zeros(size(test_cls2,1),1),ones(size(test_cls2,1),1)];

test_cls1 =[test_cls1,y_test_cls1];
test_cls2 =[test_cls2,y_test_cls2];
testData =[test_cls1;test_cls2];
%test_X=normr(testData(:,1:2)');
test_X= testData(:,1:2)';


test_t=testData(:,3:4)'; % in one versus k representation
targets =[ones(1,size(test_cls1,1)),2*ones(1,size(test_cls2,1))];

minmax_train =minmax(train_X);
no_outputnodes =2;
min_mse=1000000;
% for h1_nodes=3:7
h1_nodes=4;
h2_nodes =3;
mnet =newff(train_X,train_t,[h1_nodes,h2_nodes],{'tansig','tansig','logsig'},'traingd','learngd','mse');
mnet.trainParam.lr = 0.1;
mnet.trainParam.epochs = 10;
[net] =train(mnet,train_X,train_t,[],[]);
[mse,predict] = mlp_net_predict(net,val_X,val_t);
if mse<min_mse
    min_mse=mse;
    min_h1_nodes=h1_nodes;
    min_h2_nodes =h2_nodes;
    minnet =net;
    disp(sprintf('h1 %d h2 %d ',h1_nodes,h2_nodes))
end;
% end;
% end;
[mse,predict] = mlp_net_predict(minnet,val_X,val_t);
figure();
plotconfusion(val_t,predict);
[mse,predict] = mlp_net_predict(minnet,test_X,test_t);
  figure;
 plotconfusion(test_t,predict);
 beta=0.65;
 disp(beta);


h1_weights= minnet.IW{1,1};
h1_bias =minnet.b{1,1};

w0 = [];
for i =1:min_h1_nodes
w0  =[w0;(ones(1,size(train_X,2))*h1_bias(i))];
end;
h1_a_c1 =(train_X'*h1_weights')'+w0;
n_c1 =beta*h1_a_c1;
n1_nodes_values_c1= tansig(n_c1');

for ni =1:min_h1_nodes
figure();

plot3(train_X(1,:)',train_X(2,:)',tansig(n_c1(ni,:)),'r*');

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer 1');
title('Hidden Layer 1 - Hidden Node');
% legend('Class 1','Class 2');
end;

%%
%%Hidden Layer 2
h2_weights= minnet.LW{2,1};
h2_bias =minnet.b{2,1}
w0 = [];
for i =1:min_h2_nodes
w0  =[w0;(ones(1,size(train_X,2))*h2_bias(i))];
end;
h2_a_c1 =(n1_nodes_values_c1*h2_weights')'+w0;
n_c1 =beta*h2_a_c1;
n2_nodes_values_c1= tansig(n_c1);

for nj =1:min_h2_nodes
fig= figure();
plot3(train_X(1,:)',train_X(2,:)',tansig(n_c1(nj,:)),'r*');

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer 2');
% legend('Class 1','Class 2');
title('Hidden Layer 2- Hidden Node');
end;

[out] =sim(minnet,train_X);


fig=figure();
plot3(train_X(1,:)',train_X(2,:)',out(1,:)','r*');

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
title('Output Node 1');

fig=figure();
plot3(train_X(1,:)',train_X(2,:)',out(2,:)','r*');

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
title('Output Node 2');


disp(min_h1_nodes);
disp(min_h2_nodes);