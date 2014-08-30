function V = learn_filtVecs(patches, numFilters)
% Learn optimal values for filtVecs matrix via Stochastic Gradient Descent
	dataDim = size(patches, 2);   % d = size of patch vector
	V = randn(dataDim+1, numFilters);   % random initial values (+bias row)
	epochSize = 50;
	patchSize = length(patches);
	alpha = .1;   % changeable

    for e = 1:epochSize
        fprintf('Learning FilterVecs: epoch %d\n', e);
        patchOrder = randperm(patchSize);   % new random order of patches
        p_cost = 0;
        for t = 1:patchSize   % update filtVecs matrix one random patch at a time
            patch = [patches(patchOrder(t),:)';1];   % add bias element
            [cost, grad] = patch_grad(V, patch);
            V = V - alpha * grad;
            p_cost = p_cost + cost;
        end
        fprintf('Cost from epoch %d: %d\n', e, p_cost);
    end

function [cost, grad] = patch_grad(V, patch)
% calculate gradient of the reconstruction error w.r.t. filtVecs matrix V
	d = size(V, 1);
	sig_v = sigmoid_sae(V' * patch);
	z = V * sig_v;
	sig_z = sigmoid_sae(z);
    diff = sig_z - patch;
    cost = diff' * diff;
	u = diff .* sig_z .* (1 - sig_z) * 2;
	y = V .* repmat(sig_v' .* (1 - sig_v'), d, 1);
	grad = u * sig_v' + patch * (u' * y);