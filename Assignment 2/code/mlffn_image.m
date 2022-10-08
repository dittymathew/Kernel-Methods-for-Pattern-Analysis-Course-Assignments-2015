totaldata = importdata('data/CompleteData.mat');

%Split the Image dataset in ratio 70:10:20 for Train:Validation:Test 
G12 =[6,	14,	4,	12,	7];

%train_data = cell(5,1); % each cell contains the data for each of the class
%val_data = cell(5,1);
%test_data = cell(5,1);
train_data =[];
val_data=[];
test_data=[];
y_cls = cell(5,1);
y_cls{1} = [1,0,0,0,0];
y_cls{2} = [0,1,0,0,0];
y_cls{3} = [0,0,1,0,0];
y_cls{4} = [0,0,0,1,0];
y_cls{5} = [0,0,0,0,1];

for i = 1:size(G12,2)
    temp = totaldata{G12(i)};
    n1= round(0.7*size(temp,1));
    n2= round(0.8*size(temp,1));
    n3= size(temp,1);
    
    train_data_i = temp(1:n1,1:48);
    cls_apnd =[zeros(size(train_data_i,1),1),zeros(size(train_data_i,1),1),zeros(size(train_data_i,1),1),zeros(size(train_data_i,1),1),zeros(size(train_data_i,1),1)];
    y_cls_apnd =[];
    for row_i =1:size(cls_apnd,1)
        y_cls_apnd =[y_cls_apnd;cls_apnd(row_i,:)+y_cls{i}];
    end
    train_data_cls_i =[train_data_i,y_cls_apnd];
    train_data =[train_data;train_data_cls_i];
    
    val_data_i = temp(n1+1:n2,1:48);
    cls_apnd =[zeros(size(val_data_i,1),1),zeros(size(val_data_i,1),1),zeros(size(val_data_i,1),1),zeros(size(val_data_i,1),1),zeros(size(val_data_i,1),1)];
    y_cls_apnd =[];
    for row_i =1:size(cls_apnd,1)
        y_cls_apnd =[y_cls_apnd;cls_apnd(row_i,:)+y_cls{i}];
    end
    val_data_cls_i =[val_data_i,y_cls_apnd];
    val_data =[val_data;val_data_cls_i];
    
    test_data_i = temp(n2+1:n3,1:48);
    cls_apnd =[zeros(size(test_data_i,1),1),zeros(size(test_data_i,1),1),zeros(size(test_data_i,1),1),zeros(size(test_data_i,1),1),zeros(size(test_data_i,1),1)];
    y_cls_apnd =[];
    for row_i =1:size(cls_apnd,1)
        y_cls_apnd =[y_cls_apnd;cls_apnd(row_i,:)+y_cls{i}];
    end
    test_data_cls_i =[test_data_i,y_cls_apnd];
    test_data =[test_data;test_data_cls_i];
        
end
train_X=train_data(:,1:48)';
train_t=train_data(:,49:53)';

val_X= val_data(:,1:48)';
val_t= val_data(:,49:53)';

test_X=test_data(:,1:48)';
test_t=test_data(:,49:53)';

X_data =[train_X';val_X';test_X'];
X_norm =zscore(X_data)';
train_X=X_norm(:,1:686);
val_X= X_norm(:,687:784);
test_X =X_norm(:,785:979);

X= [6:40];
%for k =1:20
    Y=zeros(size(X));
min_mse =10000;
for no_hiddennodes =6:40
  
minmax_train =minmax(train_X);
no_outputnodes =5;
min_h1_mse =10000;
% min_h1_mse =10000;
for h2_nodes= 5:no_hiddennodes-1
%for lr =0.01:0.01:0.1
lr=0.1
mnet =newff(train_X,train_t,[no_hiddennodes,h2_nodes],{'tansig','tansig','tansig'},'trainlm','learngd','mse');
mnet.trainParam.lr = lr;
mnet.trainParam.mc=0.8;
mnet.trainParam.epochs = 500;
[net] =train(mnet,train_X,train_t,[],[]);
[mse,predict] = mlp_net_predict(net,val_X,val_t);
if mse <min_mse
    min_mse =mse;
    min_valP =predict;
    min_hiddennodes= no_hiddennodes;
    min_h2_nodes= h2_nodes;
    min_lr =lr;
    min_net =net;
    Y(1,no_hiddennodes-5)=mse;
end;


% end;

if mse<min_h1_mse
    min_h1_mse=mse;
end;
%     end;
end;
if Y(1,no_hiddennodes-5) ==0
    Y(1,no_hiddennodes-5)=min_h1_mse;
end;
end;

 perf = perform(net,predict,val_t);
 %confusionmat(val_t,predict)
 %disp(Evaluate(val_t,predict));
 figure();
 plotconfusion(val_t,min_valP);
% 
 [mse,predict] = mlp_net_predict(min_net,test_X,test_t);
  figure;
 plotconfusion(test_t,predict);
%  view(min_net);
% h1_weights= min_net.IW{1,1};
% h1_a =train_X'*h1_weights';
% n =0.05*h1_a;
% n1_nodes_values= tansig(n);
% figure();
% plot3(train_X(1,:)',train_X(2,:)',tansig(n(:,1)),'*');
% set(gca,'dataaspectratio',[1 1 1],'xgrid','on','ygrid','on')
%  yPredict =zeros(1,size(test_X,2));
%  i=1;
%  for yp =predict
%      if yp(1,1)==1
%          
%          yPredict(1,i)=1;
%      elseif yp(2,1) ==1
%          yPredict(1,i)=2;
%      end;
%      i =i+1;
%     
%  end;
%  plotboundary(test_X',min_net);
 disp(sprintf('Min no of hidden nodes =%d',min_hiddennodes));
  disp(sprintf('Min no of hidden nodes 2 =%d',min_h2_nodes));
   disp(sprintf('Min learning rate =%d',min_lr));

figure();
plot(X,Y,'LineWidth',2);
xlabel('No of Hidden Nodes');
ylabel('MSE');
title('MSE vs No of hidden nodes , No of layers =1');
