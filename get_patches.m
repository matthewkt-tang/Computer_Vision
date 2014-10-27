function allPatches = get_patches(X, rfSize, CIFAR_DIM, whitening, useSPM)
% X input image matrix [l by prod(size(CIFAR_DIM)) matrix]
% output (allPatches) is a cell of patches with same overall size as X
  % wrong: can not set whitening value here, need to pass from outside
  %whitening = false;
  numImg = size(X,1);
  
  % retrieve patches from 4 spatial areas for all training images
  % allPatches{i,1}, allPatches{i,2}, allPatches{i,3}, allPatches{i,4}
  %  for patches extracted from the top-left, bottom-left, top-right, 
  %  bottom-right of the ith image
  if useSPM,
      allPatches = cell(numImg,4);
      labelI = zeros(CIFAR_DIM(1)-rfSize+1, CIFAR_DIM(2)-rfSize+1);
      h1 = floor((CIFAR_DIM(1)-rfSize+1)/2);
      h2 = floor((CIFAR_DIM(2)-rfSize+1)/2);
      labelI(1:h1, 1:h2) = 1;	
      labelI(h1+1:end, 1:h2) = 2;
      labelI(1:h1, h2+1:end) = 3;
      labelI(h1+1:end, h2+1:end) = 4;
      labelI = labelI(:);
  else
      allPatches = cell(numImg, 1);
  end
  for i=1:numImg
    if (mod(i,1000) == 0) fprintf('Getting patches: %d / %d\n', i, size(X,1)); end
    
    % extract overlapping sub-patches into rows of 'patches'
	% patches is a 729 by 108 matrix with patch feature vectors in columns
    patches = [ im2col(reshape(X(i,1:1024),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,1025:2048),CIFAR_DIM(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2049:end),CIFAR_DIM(1:2)), [rfSize rfSize]) ]';
	%% make sure dimensions of different arrays match each other
    %assert(size(patches,1)==length(labelI));
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
	%allPatches(i) = {patches};
    if useSPM,
        for j=1:4,
            allPatches{i, j} = patches(labelI==j, :);
        end
    else
        allPatches{i, 1} = patches;
    end
  end
