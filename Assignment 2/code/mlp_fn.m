function [mse,predict] =mlp_net_predict(net,test_X,test_t)
[out] =sim(net,test_X);
size(out);
predict =zeros(size(test_t));
size(test_t)
size(predict)
for i =1:size(test_X,2)
    [m,idx] =max(out(:,i));
    predict(idx,i) =1;
   
end
mse=0
% mse = (sum(sum((test_t-predict).*(test_t-predict))));
%  Y(1,no_hiddennodes-3) =mse;
 disp(mse)

end