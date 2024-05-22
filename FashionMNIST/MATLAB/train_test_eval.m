function [accuracy, loss] = train_test_eval(trainImages, trainLabels, testImages, testLabels, layers, options)
    % Train the network
    net = trainNetwork(trainImages, trainLabels, layers, options);
    
    % Evaluate the network
    YPred = classify(net, testImages);
    accuracy = sum(YPred == testLabels) / numel(testLabels);
    loss = computeLoss(net, testImages, testLabels);  % Assuming a function to compute loss
end

function loss = computeLoss(net, images, labels)
    predictions = predict(net, images);
    loss = crossentropy(predictions, labels);
end


