clear; clc;
addpath('D:\New folder\FashionMNIST\MATLAB\models');  % Add model directory to path

% Load the dataset
[trainImages, trainLabels, testImages, testLabels] = get_data();

% Convert labels to categorical
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

% Reshape image data to include the channel dimension
trainImages = reshape(trainImages, [28, 28, 1, size(trainImages, 3)]);
testImages = reshape(testImages, [28, 28, 1, size(testImages, 3)]);

% Debugging: Display dimensions of the data
disp(['Size of training images: ', mat2str(size(trainImages))]);
disp(['Size of training labels: ', num2str(numel(trainLabels))]);
disp(['Size of test images: ', mat2str(size(testImages))]);
disp(['Size of test labels: ', num2str(numel(testLabels))]);

% Define training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Initialize the results table
results = table('Size', [3 3], ...
                'VariableTypes', {'string', 'double', 'double'}, ...
                'VariableNames', {'Model', 'Accuracy', 'Loss'});

% List of model functions
modelFuncs = {@modelV0, @modelV1, @modelV2};
modelNames = {'Model V0', 'Model V1', 'Model V2'};

% Loop through each model, train, and evaluate
for k = 1:length(modelFuncs)
    fprintf('Training and evaluating %s\n', modelNames{k});
    
    % Create the model
    layers = modelFuncs{k}(28, 10);  % Assuming input size 28x28 and 10 classes
    
    % Train and evaluate the model
    [net, info] = trainNetwork(trainImages, trainLabels, layers, options);
    
    % Evaluate the model
    YPred = classify(net, testImages);
    accuracy = sum(YPred == testLabels) / numel(testLabels);
    finalLoss = info.TrainingLoss(end);
    
    % Store results
    results.Model(k) = modelNames{k};
    results.Accuracy(k) = accuracy;
    results.Loss(k) = finalLoss;
end

% Display the results
disp(results);





