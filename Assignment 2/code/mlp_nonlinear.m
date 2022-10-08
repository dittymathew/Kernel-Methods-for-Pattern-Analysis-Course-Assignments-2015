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
X= [3:7];
%for k =1:20
    Y=zeros(size(X));
min_mse =10000;
for no_hiddennodes =3:7
  
minmax_train =minmax(train_X);
no_outputnodes =2;
mnet =newff(train_X,train_t,[no_hiddennodes,3],{'tansig','tansig','tansig'},'traingd','learngd','mse');
mnet.trainParam.lr = 0.1;
mnet.trainParam.epochs = 1000;
[net] =train(mnet,train_X,train_t,[],[]);
[mse,predict] = mlp_net_predict(net,val_X,val_t);
if mse <min_mse
    min_mse =mse;
    min_valP =predict;
    min_hiddennodes= no_hiddennodes;
    min_net =net;
end;
Y(1,no_hiddennodes-2)=mse;
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
 plotboundary(test_X',min_net);
 disp(sprintf('Min no of hidden nodes =%d',min_hiddennodes));

figure();
plot(X,Y,'LineWidth',2);
xlabel('No of Hidden Nodes');
ylabel('MSE');
title('MSE vs No of hidden nodes , No of layers =1');
%end
%%%
% mnet =newff(train_X,train_t,[min_hiddennodes],{'tansig'},'traingd','learngd','mse');
% [net,tr] =train(mnet,train_X,train_t,[],[]);
% disp(net);
%[out] =sim(net,test_X);



%%