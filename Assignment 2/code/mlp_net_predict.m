function [mse,predict] =mlp_net_predict(net,test_X,test_t)
[out] =sim(net,test_X);

predict =zeros(size(test_t));
for i =1:size(test_X,2)
    [m,idx] =max(out(:,i));
    predict(idx,i) =1;
   
end
 mse = (sum(sum((test_t-predict).*(test_t-predict))))/size(test_t,2);
%  Y(1,no_hiddennodes-3) =mse;
% disp(mse);

end