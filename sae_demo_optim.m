function sae_demo_optim
%CIFAR_DIR='/kusers/academic/matthew/cifar/';
CIFAR_DIR='../../data/cifar-10-batches-mat/';

%{
assert(~strcmp(CIFAR_DIR, '/path/to/cifar/cifar-10-batches-mat/'), ...
       ['You need to modify kmeans_demo.m so that CIFAR_DIR points to ' ...
        'your cifar-10-batches-mat directory.  You can download this ' ...
        'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);
%}    


%% Configuration
%addpath minFunc;
rfSize = 6;
%numCentroids=1600;
numFilters=100;
whitening=false;%true;
numPatches = 400;%400000;
CIFAR_DIM=[32 32 3];
biased = 1;%-1 for unbiased;
C = 100;
useSPM = 1;% 1 for using SPM of 2x2 grid, 0 for standard BOW

%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

% trim data
%{
trainSize = 500;
dl1 = trimData([f1.data f1.labels],trainSize);
dl2 = trimData([f2.data f2.labels],trainSize);
dl3 = trimData([f3.data f3.labels],trainSize);
dl4 = trimData([f4.data f4.labels],trainSize);
dl5 = trimData([f5.data f5.labels],trainSize);
f1data = dl1(:,1:end-1);
f2data = dl2(:,1:end-1);
f3data = dl3(:,1:end-1);
f4data = dl4(:,1:end-1);
f5data = dl5(:,1:end-1);
f1labels = dl1(:,end);
f2labels = dl2(:,end);
f3labels = dl3(:,end);
f4labels = dl4(:,end);
f5labels = dl5(:,end);
%}
trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!	
clear f1 f2 f3 f4 f5;

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);

% trim test data
%dl1 = trimData([f1.data f1.labels],400);
%f1data = dl1(:,1:end-1);
%f1labels = dl1(:,end);

testX = double(f1.data);
testY = double(f1.labels) + 1;
%testX = double(f1data);
%testY = double(f1labels) + 1;
%clear dl1 f1data f1labels;
clear f1;



% learn initial filtVecs by running K-means clustering
kmeansfile = 'kmeans_codebook.mat';
if whitening, kmeansfile = 'kmeans_codebook_whiten.mat';    end
if exist(kmeansfile, 'file'),
	load(kmeansfile, 'filtVecs');
else
    % extract random patches for kmeans clustering
    patches = zeros(numPatches, rfSize*rfSize*3);
    for i=1:numPatches
      if (mod(i,10000) == 0) fprintf('Extracting patch: %d / %d\n', i, numPatches); end

      r = random('unid', CIFAR_DIM(1) - rfSize + 1);
      c = random('unid', CIFAR_DIM(2) - rfSize + 1);
      patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);
      patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
      patches(i,:) = patch(:)';
    end

    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

    if (whitening)
      C = cov(patches);
      M = mean(patches);
      [V,D] = eig(C);
      P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
      patches = bsxfun(@minus, patches, M) * P;
    end
	filtVecs = run_kmeans(patches, numFilters, 50);
	% need to normalise filtVecs to unit norms
	save(kmeansfile, 'filtVecs');
end

filtVecs = [filtVecs, zeros(size(filtVecs, 1),1)];
%{
%load fv1.mat

batchSize = 5000;
disp('Extract training features');
nbatches = ceil(length(trainY)/batchSize);
if useSPM,
    featDim = size(filtVecs, 1)*4;
else
    featDim = size(filtVecs, 1); 
end
trainFV = zeros(length(trainY), featDim);
for i=1:nbatches,
    fprintf('batch %d/%d\n', i, nbatches); 
    sidx = (i-1)*batchSize + 1;
    eidx = min(i*batchSize, length(trainY)); 
    allPatches = get_patches(trainX(sidx:eidx, :), rfSize, CIFAR_DIM, whitening, useSPM); 
    trainFV(sidx:eidx, :) = extract_features_sae_p(allPatches, filtVecs, 1);        
end
disp('Train SVM');
model = lsvmtrain(trainY, sparse(trainFV), '-c 100 -s 2 -B 1'); 
disp('Extract testing features and Test SVM'); 
nbatches = ceil(length(testY)/batchSize); 
testFV = zeros(length(testY), featDim);
for i=1:nbatches,
    fprintf('batch %d/%d\n', i, nbatches); 
    sidx = (i-1)*batchSize + 1;
    eidx = min(i*batchSize, length(testY)); 
    allPatches = get_patches(testX(sidx:eidx, :), rfSize, CIFAR_DIM, whitening, useSPM); 
    testFV(sidx:eidx, :) = extract_features_sae_p(allPatches, filtVecs, 1);        
end
[~, accu] = lsvmpredict(testY, testFV, model); 
fprintf('Test accuracy = %.2f%%\n', accu); 
%}



gp_parm = struct('rfSize', rfSize, 'whitening', whitening, 'useSPM', useSPM); 
% randomly select 5000 images for training and 5000 images for validation
randidx = randperm(length(trainY));
idxtrain = randidx(1:5000);
idxval = randidx(5001:10000);
trainY = trainY([idxtrain, idxval]);


klist = [10:10:50 100]; %gamma = 1;
% train with SPM
allPatches = get_patches(trainX([idxtrain, idxval], :), CIFAR_DIM, gp_parm);
for k=klist,
    onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, k, [num2str(k) ': kmeans+spm']);    
end
load fv1
for k=klist,    
    onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, k, [num2str(k) ': optim+spm']);
end

%{
allPatches = get_patches(trainX([idxtrain, idxval], :), CIFAR_DIM, gp_parm);
onetest0(allPatches, trainY, filtVecs, 1:5000, 5001:10000, 'kmeans+spm'); 
load fv1
onetest0(allPatches, trainY, filtVecs, 1:5000, 5001:10000, 'optim+spm');
% % repeat with BOW
% allPatches = get_patches(trainX([idxtrain, idxval], :), rfSize, CIFAR_DIM, whitening, 0);
% onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, gamma, 'kmeans+bow');
% onetest(allPatches, trainY, fvMatrix', 1:5000, 5001:10000, gamma, 'optim+bow');
%}


function onetest0(allPatches, trainY, filtVecs, idx1, idx2, infostr)
C = 100;
trainFV = extract_features_sae_p(allPatches(idx1, :), filtVecs, 1);
valFV = extract_features_sae_p(allPatches(idx2, :), filtVecs, 1);
model = lsvmtrain(trainY(idx1), sparse(trainFV), ['-s 2 -c ' num2str(C/size(trainFV, 1)) ' -B 1']);
[~, accu] = lsvmpredict(trainY(idx2), valFV, model);
fprintf('%s: accuracy = %.2f%%\n', infostr, accu);



function onetest(allPatches, trainY, filtVecs, idx1, idx2, k, infostr)
C = 100;
trainFV = extract_features_sae_sp(allPatches(idx1, :), filtVecs, k);
valFV = extract_features_sae_sp(allPatches(idx2, :), filtVecs, k);
model = lsvmtrain(trainY(idx1), sparse(trainFV), ['-s 2 -c ' num2str(C/size(trainFV, 1)) ' -B 1']);
[~, accu] = lsvmpredict(trainY(idx2), valFV, model);
fprintf('%s: accuracy = %.2f%%\n', infostr, accu);


function onetest_old(allPatches, trainY, filtVecs, idx1, idx2, gamma, infostr)
C = 100;
trainFV = extract_features_sae_p(allPatches(idx1, :), filtVecs, gamma);
valFV = extract_features_sae_p(allPatches(idx2, :), filtVecs, gamma);
model = lsvmtrain(trainY(idx1), sparse(trainFV), ['-s 2 -c ' num2str(C/size(trainFV, 1)) ' -B 1']);
[~, accu] = lsvmpredict(trainY(idx2), valFV, model);
fprintf('%s: accuracy = %.2f%%\n', infostr, accu);