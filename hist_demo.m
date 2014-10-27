function hist_demo
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
useSPM = 0;% 1 for using SPM of 2x2 grid, 0 for standard BOW

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



nbins = 256; % # of bins for histogram
togray = true;  % convert to grayscale or not
% disp('Extract histogram features for training images');
% trainFV = getHistVecs(trainX, nbins, togray);
% fprintf('\n');
% disp('Train SVM');
% model = lsvmtrain(trainY, sparse(trainFV), '-c 100 -s 2 -B 1'); 
% disp('Extract testing features and Test SVM'); 
% testFV = getHistVecs(testX, nbins, togray);
% [~, accu] = lsvmpredict(testY, testFV, model); 
% fprintf('Gray hist: Test accuracy = %.2f%%\n', accu); 

togray = false;
disp('Extract histogram features for training images');
trainFV = getHistVecs(trainX, nbins, togray);
disp('Extract testing features');
testFV = getHistVecs(testX, nbins, togray);
fprintf('\n');
clist = [1 10 100 1000];

for C=clist,
disp('Train SVM');
model = lsvmtrain(trainY, sparse(trainFV), ['-c ' num2str(C) ' -s 2 -B 1']); 
disp('Test SVM'); 
[~, accu] = lsvmpredict(testY, testFV, model); 
fprintf('Color hist: Test accuracy = %.2f%%\n', accu); 
end

%{


%allPatches = get_patches(trainX, rfSize, CIFAR_DIM, whitening);
% randomly select 5000 images for training and 5000 images for validation
randidx = randperm(length(trainY));
idxtrain = randidx(1:5000);
idxval = randidx(5001:10000);

trainY = trainY([idxtrain, idxval]); 
gamma = 1;


% train with SPM
allPatches = get_patches(trainX([idxtrain, idxval], :), rfSize, CIFAR_DIM, whitening, 1);
onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, gamma, 'kmeans+spm');
load fv1
onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, gamma, 'optim+spm');
% % repeat with BOW
% allPatches = get_patches(trainX([idxtrain, idxval], :), rfSize, CIFAR_DIM, whitening, 0);
% onetest(allPatches, trainY, filtVecs, 1:5000, 5001:10000, gamma, 'kmeans+bow');
% onetest(allPatches, trainY, fvMatrix', 1:5000, 5001:10000, gamma, 'optim+bow');
%}


function trainFV = getHistVecs(trainX, nbins, togray)
if togray,
    featDim = nbins; 
else
    featDim = nbins * 3;
end
trainFV = zeros(size(trainX, 1), featDim);
for i=1:size(trainX, 1), 
    if mod(i, 1000)==0, fprintf('.');   end    
    Xi = reshape(trainX(i, :), 32, 32, 3); 
    if togray,  % compute grayscale histogram       
        Xi = rgb2gray(Xi); 
        hist_i = imhist(Xi, nbins); 
        trainFV(i, :) = hist_i(:)';
    else        % compute histgram for each
        for j=1:3,
            hist_i = imhist(Xi(:, :, j), nbins);
            trainFV(i, (j-1)*nbins+1:j*nbins) = hist_i;
        end
    end
end



function onetest(allPatches, trainY, filtVecs, idx1, idx2, gamma, infostr)
C = 100;
trainFV = extract_features_sae_p(allPatches(idx1, :), filtVecs, gamma);
valFV = extract_features_sae_p(allPatches(idx2, :), filtVecs, gamma);
model = lsvmtrain(trainY(idx1), sparse(trainFV), ['-s 2 -c ' num2str(C/size(trainFV, 1)) ' -B 1']);
[~, accu] = lsvmpredict(trainY(idx2), valFV, model);
fprintf('%s: accuracy = %.2f%%\n', infostr, accu);