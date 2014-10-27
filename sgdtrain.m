function sgdtrain(wVecs, filtVecs, allPatches, trainY, parm)
% train svm and filter vectors simultaneously using sgd algorithm
% wVecs: (4*nfilt+1)*nc vector, assuming L1 SPM that divides the image into 4 cells
% filtVecs: nfilt*(fd+1)

error('this program was not finished yet');
% set parameters
C = parm.C;
patchSize = parm.patchSize;
imgdim = parm.imgdim;
epochs = 10;%parm.epochs;
batchSize = 50;%parm.batchSize;
% data dimension checks
[nfilt, fd] = size(filtVecs);	fd = fd-1;
nc = size(wVecs, 2);	
if size(wVecs, 1)~=4*nfilt+1,
	error('wrong row dim for wVecs.');
end
if size(allPatches{1}, 1)~=fd | fd~=prod(patchSize)*imgdim(3),
	error('wrong dim for local feat descriptors.');
end
N = length(allPatches);
nbatches = ceil(N/batchSize);
for e=1:epochs,
	% randomly shuffle training points for each epoch
	randidx = randperm(N);
	allPatches = allPatches(randidx);
	trainY = trainY(randidx);
	for b=1:nbatches,
		sidx = (b-1)*batchSize+1;
		eidx = min(b*batchSize, N);
		%allPatches(sidx:eidx);
		yVecs = bsxfun(@(y,ypos)2*(y==ypos)-1, trainY(sidx:eidx), 1:size(w,2));
		[gFiltVecs, gWVecs] = l2filtsvmgrad(allPatches(sidx:eidx), yVecs);
		filtVecs = filtVecs - lambda * gFiltVecs;
		wVecs = wVecs - lambda * gWVecs;
	end
end

function [gFiltVecs, gWVecs] = l2filtsvmgrad(filtVecs, w, P, yv, C)
  %t1 = tic;
  gamma = 1;
  %filtVecs = reshape(filtVecs, fvSize(1), fvSize(2));
	[nfilt, fd] = size(filtVecs);	fd = fd-1;
	gFiltVecs = zeros(size(filtVecs));
  %trainFV = extract_features_sae_p(P, filtVecs);
	% forward propagation
	for i=1:length(P),
		P{i}
	end
  trainFV = [trainFV, ones(size(trainFV, 1), 1)];
  pr = trainFV * w;
  numImg = size(P,1);
  FminusY = pr - yv;
  FminusY(pr .* yv > 1) = 0;
  loss = .5 * sum(sum(w .^ 2)) + C * mean(sum(FminusY .^ 2, 2));
  for k = 1:fvSize(1)
    % calculate delta(Q)/delta(vk) for each image i
    ds = zeros(numImg, fvSize(2));
    for i = 1:numImg
      patches = [P{i} ones(size(P{1},1),1)];
	  % patches: m x d, filtVecs: n x d
	  gk = sigmoid_sae(patches * filtVecs(k,:)');   % m x 1
	  one_minus_gk = 1 - gk;
      % ds: numImg x d
	  ds(i,:) = (gk .* one_minus_gk)' * patches * gamma / size(patches,1);
    end
    % w: n x numClass, FminusY: numImg x numClass, gFiltVecs: n x d
	gFiltVecs(k,:) = w(k,:) * FminusY' * ds * C / numImg;
  end
