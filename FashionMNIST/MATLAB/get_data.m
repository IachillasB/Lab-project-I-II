function [trainImages, trainLabels, testImages, testLabels] = get_data()
    % Define URLs for the dataset
    urls = {
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    };

    % Define the directory to store the data
    dataDir = fullfile('data', 'FashionMNIST');
    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    % Download and extract files
    for i = 1:length(urls)
        [~, name, ext] = fileparts(urls{i});
        compressedFilename = fullfile(dataDir, [name, ext]);
        if ~exist(compressedFilename, 'file')
            websave(compressedFilename, urls{i});
        end
        gunzip(compressedFilename, dataDir);
    end

    % Load the data using a helper function to read IDX format
    trainImages = readIDX(fullfile(dataDir, 'train-images-idx3-ubyte'));
    trainLabels = readIDX(fullfile(dataDir, 'train-labels-idx1-ubyte'));
    testImages = readIDX(fullfile(dataDir, 't10k-images-idx3-ubyte'));
    testLabels = readIDX(fullfile(dataDir, 't10k-labels-idx1-ubyte'));
end

function data = readIDX(filename)
    fid = fopen(filename, 'rb');
    assert(fid ~= -1, ['Could not open ', filename, '']);
    
    magicNum = fread(fid, 1, 'int32', 0, 'ieee-be');
    assert(magicNum == 2051 || magicNum == 2049, 'Bad magic number in IDX file');
    
    numItems = fread(fid, 1, 'int32', 0, 'ieee-be');
    if magicNum == 2051
        numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
        numCols = fread(fid, 1, 'int32', 0, 'ieee-be');
        data = fread(fid, numRows*numCols*numItems, 'unsigned char');
        data = reshape(data, numCols, numRows, numItems);
        data = permute(data, [2 1 3]);
    else
        data = fread(fid, numItems, 'unsigned char');
    end
    fclose(fid);
end


