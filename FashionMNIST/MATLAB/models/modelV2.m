% modelV2.m
function layers = modelV2(inputSize, numClasses)
    layers = [
        imageInputLayer([inputSize inputSize 1])
        convolution2dLayer(3, 32, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        convolution2dLayer(3, 64, 'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride',2)
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
end