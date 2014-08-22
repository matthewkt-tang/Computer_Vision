CIFAR_DIR='/kusers/academic/matthew/cifar/';

%{
assert(~strcmp(CIFAR_DIR, '/path/to/cifar/cifar-10-batches-mat/'), ...
       ['You need to modify kmeans_demo.m so that CIFAR_DIR points to ' ...
        'your cifar-10-batches-mat directory.  You can download this ' ...
        'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);
%}    

%CIFAR_DIR = 'D:/data/cifar-10-batches-mat/';

%% Configuration
addpath minFunc;
rfSize = 6;
%numCentroids=1600;
numFilters=100;
whitening=true;
numPatches = 400000;
CIFAR_DIM=[32 32 3];
biased = true;

if exist([CIFAR_DIR '/train.mat'], 'file')
	load([CIFAR_DIR '/train.mat']);
else
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
	%trainX = double([f1data; f2data; f3data; f4data; f5data]);
	%trainY = double([f1labels; f2labels; f3labels; f4labels; f5labels]) + 1; % add 1 to labels!
	%clear dl1 dl2 dl3 dl4 dl5 f1data f2data f3data f4data f5data f1labels f2labels f3labels f4labels f5labels;
	clear f1 f2 f3 f4 f5;
	save([CIFAR_DIR '/train.mat'], 'trainX', 'trainY');
end

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

% extract random patches
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

% whiten
if (whitening)
  C = cov(patches);
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
  patches = bsxfun(@minus, patches, M) * P;
end

% learn initial filtVecs by running K-means clustering
filtVecs = run_kmeans(patches, numCentroids, 50);

allPatches = get_patches(trainX, numFilters, rfSize, CIFAR_DIM);
%save([CIFAR_DIR 'patches.mat'], 'allPatches');
trainFV = extract_features_sae_p(allPatches, filtVecs);
if biased, 
    trainFV = [trainFV, ones(size(trainFV, 1), 1)];
end
%{
% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

% train classifier using SVM
C = 10;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
%}

C = 100;

maxIters = 10;
wVecs = zeros(size(trainFV,2)*max(trainY), 1);
for t = 1:maxIters
    % fix filterVecs, train wVecs
	wVecs = train_svm(trainFV, trainY, C, wVecs(:));
    % test based on current filtVecs & wVecs
    test_filtVecs(testX, testY, filtVecs, wVecs, rfSize, CIFAR_DIM, biased);
    % fix wVecs, solve filterVecs
	%pred = trainFV * wVecs;
    fprintf('Training filtVecs...\n');
    if exist([CIFAR_DIR '/fv' num2str(t) '.mat'], 'file')
        load([CIFAR_DIR '/fv' num2str(t) '.mat']);
    else
        filtVecs = train_filtVecs(allPatches, trainY, C, wVecs, filtVecs, rfSize, CIFAR_DIM);
        save([CIFAR_DIR '/fv' num2str(t) '.mat'], 'filtVecs');
    end
	% after getting filtVecs, need to recompute trainFV
    trainFV = extract_features_sae_p(allPatches, filtVecs);
    if biased, 
        trainFV = [trainFV, ones(size(trainFV, 1), 1)];
    end
    fprintf('Finished iteration %d\n', t);
end

[val,labels] = max(trainFV * wVecs, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%
%{
% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
%}
% testFV = extract_features_sae(testX, filtVecs, rfSize, CIFAR_DIM);
% if biased, 
%     testFV = [testFV, ones(size(testFV, 1), 1)];
% end
% [val,labels] = max(testFV*wVecs, [], 2);
% 
% fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

% test based on final filtVecs & wVecs
test_filtVecs(testX, testY, filtVecs, wVecs, rfSize, CIFAR_DIM, biased);