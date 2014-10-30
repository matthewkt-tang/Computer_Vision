function allPatches = get_patches(X, imdim, parm)
% Input: X [Nxd]: input images in rows
%		imdim [3-vector]: image dimesion	 		   
%	 		X input image matrix [l by prod(size(imdim)) matrix]
%		parm.	whitening
%				useSPM
%				rfSize
%				normtype
% output (allPatches) is a cell of patches with same overall size as X

    % set parameters
	whitening = getparam(parm, 'whitening', 0);
	useSPM = getparam(parm, 'useSPM', 1);
	rfSize = getparam(parm, 'rfSize', 6);
	% type of normalisation, 1 for unit norm, 2 for unit std for each variable
	normtype = getparam(parm, 'normtype', 2);
	assert(normtype==1 | normtype ==2);
  numImg = size(X,1);
  
  % retrieve patches from 4 spatial areas for all training images
  % allPatches{i,1}, allPatches{i,2}, allPatches{i,3}, allPatches{i,4}
  %  for patches extracted from the top-left, bottom-left, top-right, 
  %  bottom-right of the ith image
  if useSPM,
      allPatches = cell(numImg,4);
      labelI = zeros(imdim(1)-rfSize+1, imdim(2)-rfSize+1);
      h1 = floor((imdim(1)-rfSize+1)/2);
      h2 = floor((imdim(2)-rfSize+1)/2);
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
    patches = [ im2col(reshape(X(i,1:1024),imdim(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,1025:2048),imdim(1:2)), [rfSize rfSize]) ;
                im2col(reshape(X(i,2049:end),imdim(1:2)), [rfSize rfSize]) ]';
	%% make sure dimensions of different arrays match each other
    %assert(size(patches,1)==length(labelI));
    % do preprocessing for each patch
    
    % normalize for contrast
	if normtype==1,
		patches = bsxfun(@minus, patches, mean(patches,2));
		patches = bsxfun(@times, patches, 1./(sqrt(sum(patches.^2, 2))+1e-6));
	else
		patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
	end
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
