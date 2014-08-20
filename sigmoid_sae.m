function g = sigmoid_sae(x, gamma)
% calculate the sigmoid function
if nargin < 2
    gamma = 1;
end
g = 1 ./ (1 + exp(-gamma * x));
