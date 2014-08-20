function new_filtVecs = train_filtVecs(allPatches, trainY, C, w, filtVecs, rfSize, dim)

  fvSize = [size(filtVecs,1), rfSize * rfSize * dim(3)];
  %yVecs = convert trainY to txC matrix with 1 & -1 as values
  yVecs = bsxfun(@(y,ypos) 2*(y==ypos)-1, trainY, 1:size(w,2));

  new_filtVecs = minFunc(@my_l2filtVecsLoss, filtVecs(:), struct('MaxIter', 100, 'MaxFunEvals', 100), ...
              allPatches, yVecs, C, w, fvSize);

  new_filtVecs = reshape(new_filtVecs, fvSize(1), fvSize(2));

% 1-vs-all L2-filtVecs loss function;  similar to LibLinear.
function [loss, grad] = my_l2filtVecsLoss(filtVecs, P, yv, C, w, fvSize)
  t1 = tic;
  gamma = 1;
  filtVecs = reshape(filtVecs, fvSize(1), fvSize(2));
  trainFV = extract_features_sae_p(P, filtVecs);
  trainFV = [trainFV, ones(size(trainFV, 1), 1)];
  pr = trainFV * w;
  grad = zeros(fvSize(1), fvSize(2));
  numImg = size(P,1);
  FminusY = pr - yv;
  FminusY(find(pr .* yv > 1)) = 0;
  loss = .5 * sum(sum(w .^ 2)) + C * mean(sum(FminusY .^ 2, 2));
  for k = 1:fvSize(1)
    % calculate delta(Q)/delta(vk) for each image i
    for i = 1:numImg
      patches = P{i};
	  % patches: m x d, filtVecs: n x d
	  gk = sigmoid_sae(patches * filtVecs(k,:)');   % m x 1
	  one_minus_gk = 1 - gk;
      % ds: numImg x d
	  ds(i,:) = (gk .* one_minus_gk)' * patches * gamma / size(patches,1);
    end
    % w: n x numClass, FminusY: numImg x numClass, grad: n x d
	grad(k,:) = w(k,:) * FminusY' * ds * C / numImg;
  end
  t2 = toc(t1);
  %fprintf('loss = %.4f, (%.4f sec)\n', loss, t2);
  fprintf('loss = %.3f\n', loss);
  grad = grad(:);