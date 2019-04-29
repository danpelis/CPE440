

trainingSetPath = fullfile(pwd,'trainingData');
testSetPath = fullfile(pwd,'testData');

trainingSet = imageDatastore(trainingSetPath,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(testSetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


countEachLabel(trainingSet)
countEachLabel(testSet)

figure;

subplot(2,3,1);
imshow(trainingSet.Files{102});

subplot(2,3,2);
imshow(trainingSet.Files{304});

subplot(2,3,3);
imshow(trainingSet.Files{809});

subplot(2,3,4);
imshow(testSet.Files{1});

subplot(2,3,5);
imshow(testSet.Files{4});

subplot(2,3,6);
imshow(testSet.Files{6});


exTestImage = readimage(testSet,5);
processedImage = imbinarize(rgb2gray(exTestImage));

figure;

subplot(1,2,1)
imshow(exTestImage)

subplot(1,2,2)
imshow(processedImage)


img = readimage(trainingSet, 206);

% Extract HOG features and HOG visualization
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
[hog_32x32, vis32x32] = extractHOGFeatures(img,'CellSize',[32 32]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
subplot(2,3,4);  
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

subplot(2,3,5);
plot(vis16x16); 
title({'CellSize = [16 16]'; ['Length = ' num2str(length(hog_16x16))]});

subplot(2,3,6);
plot(vis32x32); 
title({'CellSize = [32 32]'; ['Length = ' num2str(length(hog_32x32))]});


cellSize = [16 16];
hogFeatureSize = length(hog_16x16);

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
X = trainingFeatures;
Y = trainingLabels;
t = templateSVM('Standardize',true,'KernelFunction','gaussian');

%classifier = fitcecoc(trainingFeatures, trainingLabels);
Mdl = fitcecoc(X,Y,'Learners',t,'FitPosterior',true,...
    'ClassNames',{'16QAM','32QAM','64QAM'},...
    'Verbose',2);
