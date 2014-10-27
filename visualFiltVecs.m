function visualFiltVecs(filtVecs)
% code to visualize filt vectors
[numFilt, filtDim] = size(filtVecs); 
assert(filtDim==6*6*3); 
% a scaling to shift values to 0 and 1
maxv = max(filtVecs(:)); % maybe pick the second or third largest value would be better
minv = min(filtVecs(:)); 
filtVecs = (filtVecs - minv)/(maxv-minv);

fac = 4;

for i=1:numFilt,
    % reshape the ith row to an image
    im = reshape(filtVecs(i, :), 6, 6, 3);
    % enlarge the image using kronecker produce by a factor of 4   
    for j=1:3,
        im_out(:, :, j) = kron(im(:, :, j), ones(fac)); 
    end
    subplot(10, 10, i), imshow(im_out);     
end
