function XC = extract_features_sae_q(allPatches, filtVecs)
% allPatches input patches [l by prod(size(CIFAR_DIM)) matrix]
% XC output feature matrix
  numImages = size(allPatches,1);
  numFilters = size(filtVecs,1);
  numPatches = size(allPatches{1});
  CIFAR_DIM = [32 32];
  rfSize = 6;
  prows = CIFAR_DIM(1)-rfSize+1;
  pcols = CIFAR_DIM(2)-rfSize+1;
  halfr = round(prows/2);
  halfc = round(pcols/2);
  
  % compute features for all training images
  XC = zeros(numImages, numFilters*4);
  for i=1:numImages
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, numImages); end

	% patches matrix 729 by 108 matrix, m = 729
	% patches = [x_{i,1}; x_{i,2};... ;x_{i,m}]
	% pass function g to each patch
	% i.e. calculating g_k(x_{i,j}) for each j 
	% purpose: calculate feature vector for image i (s_i defined in the paper)
	% begin work here
	patches = sigmoid_sae([allPatches{i} ones(numPatches,1)] * filtVecs'];
	%XC(i,:) = [];

	% quadrant calculation
	patches = reshape(patches, prows, pcols, numFilters);

	% pool over quadrants
	q1 = mean(mean(patches(1:halfr, 1:halfc, :), 1),2);
	q2 = mean(mean(patches(halfr+1:end, 1:halfc, :), 1),2);
	q3 = mean(mean(patches(1:halfr, halfc+1:end, :), 1),2);
	q4 = mean(mean(patches(halfr+1:end, halfc+1:end, :), 1),2);
    
	% concatenate into feature vector
	XC(i,:) = [q1(:);q2(:);q3(:);q4(:)]';
  end
