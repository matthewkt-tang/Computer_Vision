function XC = extract_features_sae(X, filtVecs, rfSize, CIFAR_DIM)
% X input image matrix [l by prod(size(CIFAR_DIM)) matrix]
% XC output feature matrix
  numFilters = size(filtVecs,1);
  whitening = false;
  
  % compute features for all training images
  XC = zeros(size(X,1), numFilters);
  for i=1:size(X,1)
    if (mod(i,1000) == 0) fprintf('Extracting features: %d / %d\n', i, size(X,1)); end
    
    % extract overlapping sub-patches into rows of 'patches'
	% patches is a 729 by 108 matrix with patch feature vectors in columns
    patches = [ im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';
    % do preprocessing for each patch
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    % whiten
    if (whitening)
      C = cov(patches);
      M = mean(patches);
      [V,D] = eig(C);
      P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
      patches = bsxfun(@minus, patches, M) * P;
    end
    
    
	% patches matrix 729 by 108 matrix, m = 729
	% patches = [x_{i,1}; x_{i,2};... ;x_{i,m}]
	% pass function g to each patch
    % i.e. calculating g_k(x_{i,j}) for each j
	% purpose: calculate feature vector for image i (s_i defined in the paper)
	% begin work here
	XC(i,:) = mean(sigmoid_sae(patches * filtVecs'));
    %XC(i,:) = [];
	% stop here
  end
