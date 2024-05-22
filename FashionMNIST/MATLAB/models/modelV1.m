% modelV1.m
function layers = modelV1(inputSize, numClasses)
    layers = [
        imageInputLayer([inputSize inputSize 1])
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
end