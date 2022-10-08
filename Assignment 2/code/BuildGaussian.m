function [mu covar] =BuildGaussian(data)
mu = mean(data);
covar = cov(data);





