function test_filtVecs(testX, testY, filtVecs, wVecs, rfSize, dim, biased)
% Print and return accuracy on test data for certain filtVecs
testFV = extract_features_sae_test(testX, filtVecs, rfSize, dim);
if biased, 
    testFV = [testFV, ones(size(testFV, 1), 1)];
end
[val,labels] = max(testFV*wVecs, [], 2);

fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));

end

