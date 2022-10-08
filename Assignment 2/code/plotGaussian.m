function plotGaussian(X,model)
plotContour(X,model);
plotboundary(X,model);
end
 
function plotboundary(X,model)

xmin = min(X(:,1));
xmax = max(X(:,1));

ymin = min(X(:,2));
ymax = max(X(:,2));

xrange = [xmin xmax];
yrange = [ymin ymax];

inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
image_size = size(x);
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
surfclasses = BayesianClassify(model,xy);
[idx]  = surfclasses;
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

m=size(X,1);
n=m/3;
for i=1:m
    if(i<=n)
        scatter(X(i,1),X(i,2),15,[0 0 1]);   %scatter(X,Y,S,C) - x cord, y cord, size of marker circle,color of marker (RGB)
    end
    if(i<=2*n && i>n)
        scatter(X(i,1),X(i,2),15,[1 0 0]);
    end
    if(i<=3*n && i>2*n)
        scatter(X(i,1),X(i,2),15,[0 1 0]);
    end
%     if(i<=4*n && i>3*n)
%         scatter(X(i,1),X(i,2),5,[0 1 1]);
%     end
    %hold on;
    
end

hold off;
end

function plotContour(X,model)
xmin = min(X(:,1));
xmax = max(X(:,1));

ymin = min(X(:,2));
ymax = max(X(:,2));

xrange = [xmin xmax];
yrange = [ymin ymax];

inc = 0.5;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
image_size = size(x);
%z = zeros(image_size);

xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
%xy_size = size(xy);


%surfclasses = BayesianClassify(model,xy);

%for i = 1:xy_size(1)
   % if (surfclasses(i) == 1)
         mu=model{1,1};
         sigma=model{1,2};
         [v1 d1]=eig(sigma);
         z1=mvnpdf(xy,mu,sigma);
%         z(i) = -0.5*(xy(i,1)-mu)*pinv(sigma)*(xy(i,2)-mu)'-0.5*log(det(sigma));
    %elseif(surfclasses(i) == 1)
         mu=model{2,1};
         sigma=model{2,2};
         [v2 d2]=eig(sigma);
         z2=mvnpdf(xy,mu,sigma);
%         z(i) = -0.5*(xy(i,1)-mu)*pinv(sigma)*(xy(i,2)-mu)'-0.5*log(det(sigma));
    %elseif(surfclasses(i) == 1)
         mu=model{3,1};
         sigma=model{3,2};
         [v3 d3]=eig(sigma);
         z3=mvnpdf(xy,mu,sigma);
%         z(i) = -0.5*(xy(i,1)-mu)*pinv(sigma)*(xy(i,2)-mu)'-0.5*log(det(sigma));
    %elseif(surfclasses(i) == 1)
       %  mu=model{4,1};
       %  sigma=model{4,2};
       %  [v4 d4]=eig(sigma);
       %  z4=mvnpdf(xy,mu,sigma);
%         z(i) = -0.5*(xy(i,1)-mu)*pinv(sigma)*(xy(i,2)-mu)'-0.5*log(det(sigma));
    %end
%end

 %[idx]  = surfclasses;
 %decisionmap = reshape(idx ,image_size);
 z1=reshape(z1,image_size(1),image_size(2));
z2=reshape(z2,image_size(1),image_size(2));
z3=reshape(z3,image_size(1),image_size(2));
%z4=reshape(z4,image_size(1),image_size(2));

figure;
title('Plot of the gaussian functions');
hold on;
surf(x,y,z1+0.05);
surf(x,y,z2+0.05);
surf(x,y,z3+0.05);
%surf(x,y,z4+0.05);


contour(x,y,z1);
contour(x,y,z2);
contour(x,y,z3);

%contour(x,y,z4);


view(45,30);
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
%cmap = [1 0.9 0.9; 0.95 1 0.95; 0.9 0.9 1; 0.5 0.5 0.5];
grid on;

xlabel('x');
ylabel('y');
zlabel('z');

hold off;

figure;hold on;
title('Contour Map');
contour(x,y,z1);
contour(x,y,z2);
contour(x,y,z3);


hold off;
axis equal;

end

