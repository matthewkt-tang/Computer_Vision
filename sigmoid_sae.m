function g = sigmoid_sae(x, gamma, sparse)
% calculate the sigmoid function
if nargin < 2
    gamma = 1;
end
if nargin < 3
    sparse = false;
end
g = 1 ./ (1 + exp(-gamma * x));

if sparse
    % apply 20% row sparsity to the input matrix g
    g = g .* (g >= prctile(g,80,2)*ones(1,size(g,2)));
end