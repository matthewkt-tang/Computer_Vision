function allPatches = get_patches(X, numFilters, rfSize, CIFAR_DIM)
% X input image matrix [l by prod(size(CIFAR_DIM)) matrix]
% output (allPatches) is a cell of patches with same overall size as X
  whitening = false;
  numImg = size(X,1);
  
  % retrieve patches for all training images
  allPatches = cell(numImg,1);
  for i=1:numImg
    if (mod(i,1000) == 0) fprintf('Getting patches: %d / %d\n', i, size(X,1)); end
    
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
	allPatches(i) = {patches};
  end