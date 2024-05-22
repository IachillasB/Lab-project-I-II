% modelV0.m
function layers = modelV0(inputSize, numClasses)
    layers = [
        imageInputLayer([inputSize inputSize 1])
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer];
end