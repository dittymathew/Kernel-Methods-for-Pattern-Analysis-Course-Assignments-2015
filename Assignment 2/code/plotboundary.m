
function plotboundary(X,model,t)
size(X);
xmin = min(X(:,1));
xmax = max(X(:,1));

ymin = min(X(:,2));
ymax = max(X(:,2));

xrange = [xmin xmax];
yrange = [ymin ymax];
%disp(sprintf('xmin %f, xmax %f ymin %f ymax %f ',xmin,xmax,ymin,ymax));
inc = 0.01;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
image_size = size(x);

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
surfclasses = nnetClassify(xy',model);
%surfclasses = BayesianClassify(model,xy);
[idx]  = surfclasses;
 

size(idx);
%disp(image_size)
decisionmap = reshape(idx ,image_size);
figure;
imagesc(xrange,yrange,decisionmap);
set(gca,'ydir','normal');
xlabel('X');
ylabel('Y');
title('Decision region plot');
hold on;
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.9 0.9; 0.95 1 0.95; 0.9 0.9 1; 0.5 0.5 0.5];
colormap(cmap);
size(X);
dataclassify= nnetClassify(X',model);

m=size(X,1);
for i=1:m
 disp(dataclassify(i,1));
 if(dataclassify(i,1) ==1)
     scatter(X(i,1),X(i,2),30,[0,0,1]);
 elseif(dataclassify(i,1) ==2)
     scatter(X(i,1),X(i,2),30,[1,0,0]);
 elseif(dataclassify(i,1) ==3)
     scatter(X(i,1),X(i,2),30,[0,1,0]);
 end;
end

hold off;
end

function [surfclasses] =  nnetClassify(X,model)
[X_predict] =sim(model,X);
size(X_predict);
surfclasses =zeros(size(X,1),1);
 i=1;
 for i=1:size(X_predict,2)
     yp =X_predict(:,i);
     
%      disp(sprintf('yp=%d',yp));
     [m,idx] =max(yp);
     surfclasses(i,1) =idx;
%      if yp(1,1)>yp(2,1)
%          
%          surfclasses(i,1)=1;
%      else
%          surfclasses(i,1)=2;
     %else:
       %  disp(yp)
    % end;
     i =i+1;
    
 end
end

