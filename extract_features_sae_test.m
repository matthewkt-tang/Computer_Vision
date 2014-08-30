function XC = extract_features_sae_test(X, filtVecs, rfSize, CIFAR_DIM)
% X input image matrix [l by prod(size(CIFAR_DIM)) matrix]
% XC output feature matrix
  numFilters = size(filtVecs,1);
  whitening = false;
  prows = CIFAR_DIM(1)-rfSize+1;
  pcols = CIFAR_DIM(2)-rfSize+1;
  halfr = round(prows/2);
  halfc = round(pcols/2);

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

	patches = sigmoid_sae([patches ones(size(patches,1),1)] * filtVecs'];
	%XC(i,:) = [];jj

	% quadrant calculation
	patches = reshape(patches, prows, pcols, numFilters);

	% pool over quadrants
	q1 = mean(mean(patches(1:halfr, 1:halfc, :), 1),2);
	q2 = mean(mean(patches(halfr+1:end, 1:halfc, :), 1),2);
	q3 = mean(mean(patches(1:halfr, halfc+1:end, :), 1),2);
	q4 = mean(mean(patches(halfr+1:end, halfc+1:end, :), 1),2);
    
	% concatenate into feature vector
	XC(i,:) = [q1(:);q2(:);q3(:);q4(:)]';

    % stop here
  end
