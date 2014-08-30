function XC = extract_features_sae_p(allPatches, filtVecs)
% allPatches input patches [l by prod(size(CIFAR_DIM)) matrix]
% XC output feature matrix
  numImages = size(allPatches,1);
  numFilters = size(filtVecs,1);
  numPatches = size(allPatches{1},1);
  
  % compute features for all training images
  XC = zeros(numImages, numFilters);
  for i=1:numImages
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, numImages); end
    
	% patches matrix 729 by 108 matrix, m = 729
	% patches = [x_{i,1}; x_{i,2};... ;x_{i,m}]
	% pass function g to each patch
    % i.e. calculating g_k(x_{i,j}) for each j 
	% purpose: calculate feature vector for image i (s_i defined in the paper)
	% begin work here
	XC(i,:) = mean(sigmoid_sae([allPatches{i} ones(numPatches,1)] * filtVecs'));
    %XC(i,:) = [];
	% stop here
  end