train_cls1 =load('data/nonlinearlySeparable/group12/class1_train.txt');
train_cls2 =load('data/nonlinearlySeparable/group12/class2_train.txt');
y_train_cls1= [ones(size(train_cls1,1),1),zeros(size(train_cls1,1),1)];
y_train_cls2= [zeros(size(train_cls2,1),1),ones(size(train_cls2,1),1)];

train_cls1 =[train_cls1,y_train_cls1];
train_cls2 =[train_cls2,y_train_cls2];

trainData =[train_cls1;train_cls2];
train_X=trainData(:,1:2)';
train_t=trainData(:,3:4)';

xmin = min(train_X(1,:));
xmax = max(train_X(1,:));

ymin = min(train_X(2,:));
ymax = max(train_X(2,:));

xrange = [xmin xmax];
yrange = [ymin ymax];

inc = 0.5;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
image_size = size(x);
%z = zeros(image_size);

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
train_new=xy;

minmax_train =minmax(train_X);
no_outputnodes =2;

% for h1_nodes=3:7
h1_nodes=4;
h2_nodes =3;
mnet =newff(train_X,train_t,[h1_nodes,h2_nodes],{'tansig','tansig','logsig'},'traingd','learngd','mse');
mnet.trainParam.lr = 0.1;
mnet.trainParam.epochs = 100;
[minnet] =train(mnet,train_X,train_t,[],[]);

beta=0.5;
 disp(beta);


h1_weights= minnet.IW{1,1};
h1_bias =minnet.b{1,1};

w0 = [];
for i =1:h1_nodes
w0  =[w0;(ones(1,size(train_new',2))*h1_bias(i))];
end;
h1_a_c1 =(train_new*h1_weights')'+w0;
n_c1 =beta*h1_a_c1;
n1_nodes_values_c1= tansig(n_c1');


for ni =1:h1_nodes
figure();
z = tansig(n_c1(ni,:));
z=reshape(z,image_size(1),image_size(2));

surf(x,y,z);

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
for i =1:h2_nodes
w0  =[w0;(ones(1,size(train_new',2))*h2_bias(i))];
end;
h2_a_c1 =(n1_nodes_values_c1*h2_weights')'+w0;
n_c1 =beta*h2_a_c1;
n2_nodes_values_c1= tansig(n_c1);

for nj =1:h2_nodes
fig= figure();
z = tansig(n_c1(nj,:));
z=reshape(z,image_size(1),image_size(2));

surf(x,y,z);


set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Hidden layer 2');
% legend('Class 1','Class 2');
title('Hidden Layer 2- Hidden Node');
end;

[out] =sim(minnet,train_new');


fig=figure();
z = out(1,:)';
z=reshape(z,image_size(1),image_size(2));

surf(x,y,z);

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
title('Output Node 1');

fig=figure();
z = out(2,:)';
z=reshape(z,image_size(1),image_size(2));

surf(x,y,z);

set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
xlabel('feature 1');
ylabel('feature 2');
zlabel('Output layer');
title('Output Node 2');


