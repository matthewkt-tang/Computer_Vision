function V = learn_filtVecs(patches, numFilters)
% Learn optimal values for filtVecs matrix via Stochastic Gradient Descent
	dataDim = size(patches, 2);   % d = size of patch vector
	V = randn(dataDim, numFilters);   % initial values
	epochSize = 10;
	patchSize = length(patches);
	alpha = 1;   % changeable
	
	for e = 1:epochSize
		fprintf('Learning FilterVecs: epoch %d\n', e);
		patchOrder = randperm(patchSize);   % new random order of patches
		for t = 1:patchSize   % update filtVecs matrix one random patch at a time
			grad = patch_grad(V, patches(patchOrder(t),:)');
			V = V - alpha * grad;
        end
        fprintf('Final gradient magnitude squared: %d\n', sum(sum(grad .^ 2)));
	end

function grad = patch_grad(V, patch)
% calculate gradient of the reconstruction error w.r.t. filtVecs matrix V
	d = size(V, 1);
	sig_v = sigmoid_sae(V' * patch);
	z = V * sig_v;
	sig_z = sigmoid_sae(z);
	u = (sig_z - patch) .* sig_z .* (1 - sig_z) * 2;
	y = V .* repmat(sig_v' .* (1 - sig_v'), d, 1);
	grad = u * sig_v' + patch * (u' * y);