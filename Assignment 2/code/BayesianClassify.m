function [classLabels] = BayesianClassify(model, testData)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
m=size(testData,1);
n=size(model,1);
disc=cell(n,1);
large=-10000000;


    %sigma1=zeros(m,n);
    for i=1:n
        mu=model{i,1};
        sigma=model{i,2};
        %disc{i,1}=((pinv(sigma)*mu')'*testData')-0.5*mu*(pinv(sigma))*mu';
        disc{i,1}=zeros(1,m);
        for k=1:m
        %disc{i,1}(1,k)=testData(k,:)*(-0.5*pinv(sigma)*testData(k,:)')+(pinv(sigma)*mu')'*testData(k,:)'-0.5*mu*(pinv(sigma))*mu'-0.5*log(det(sigma));
        disc{i,1}(1,k)=(-0.5*(testData(k,:)-mu)*pinv(sigma)*(testData(k,:)-mu)'-0.5*log(det(sigma)));%this matrix is a 1*m matrix
        end
    end
avg=mean(disc{1,1})+mean(disc{2,1})+mean(disc{3,1});
avg=avg/3;
varn=sqrt(var(disc{1,1}))+sqrt(var(disc{2,1}))+sqrt(var(disc{3,1}));
varn=varn/3;
% for i=1:n
%    for j=1:m
%        disc{i,1}(1,j)=(disc{i,1}(1,j)-avg)/varn;
%    end
% end
% for i=1:n
%     disc{i,1}=zscore(disc{i,1});
% end

    
classLabels=zeros(m,1);
            for j=1:m %m is number of test datasets
                 for i=1:n %n is number of classes
            sampleDisc=disc{i,1}(1,j);
                if(sampleDisc>large)
                    label=i;
                    large=sampleDisc;
                end
            end
            classLabels(j,1)=label;
            large=-100000000;
          end
%    maxm=max(disc{3,1});
%     minm=min(disc{3,1});
%     limit=(maxm-minm)/1000;
%     threshold=maxm-limit;
%     tp=0;
%     fp=0;
%     rocPlot(1,1)=0;
%     rocPlot(1,2)=0;
%     for x=1:999
%     for i=201 : 100
%         if(disc{3,1}(1,i)>threshold && classLabels(i,1)==3)
%             tp=tp+1;
%         end
%     end
%     
%     for i=101 : 200
%         if(disc{3,1}(1,i)>threshold && classLabels(i,1)==2)
%             fp=fp+1;
%         end
%     end
%     for i=1 : 100
%         if(disc{3,1}(1,i)>threshold && classLabels(i,1)==1)
%             fp=fp+1;
%         end
%     end
%     
%     rocPlot(x+1,2)=tp/500;
%     rocPlot(x+1,1)=fp/1000;
%     fp=0;
%     tp=0;
%     threshold=threshold-limit;
%     end
%     rocPlot(x+1,1)=1;
%     rocPlot(x+1,2)=1;
%     rocPlot
%     
%     plot(rocPlot(:,1),rocPlot(:,2));
%     title('ROC curve for class 3, dataset 4 with covariance matrix different (Bayesian) without Normalization');
%     xlabel('False Positive Rate');
%     ylabel('True Positive Rate'); 

end

