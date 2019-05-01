%{
-----------------------------------------
--------Load the training images---------
-----------------------------------------
%}

pwd = '';

trainingSetPath = fullfile(pwd,'trainingData');
testSetPath = fullfile(pwd,'testData');

imds = imageDatastore(trainingSetPath,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%testSet = imageDatastore(testSetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%{
-----------------------------------------
----------Train the classifier-----------
-----------------------------------------
%}

% Split the training images into training and test sets
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');

% Extract the labels from the images
bag = bagOfFeatures(imds);

% Train the classifier with the training sets
categoryClassifier = trainImageCategoryClassifier(imds,bag);

% Tests the classifier using the test images and displays the confusion
% matrix
confMatrix = evaluate(categoryClassifier,testSet);

% Finds the average accuracy of the classifier
mean(diag(confMatrix));

%{
-----------------------------------------
---Test the designated training images---
-----------------------------------------
%}

% Testing  16QAM test images
srcFiles = dir('testData\16QAM');
pwd = 'testData\16QAM';
for i = 3 : length(srcFiles)
filename = strcat('testData\16QAM\',srcFiles(i).name);
img = imread(fullfile(pwd,srcFiles(i).name));
[labelIdx, score] = predict(categoryClassifier,img);
srcFiles(i).name;
categoryClassifier.Labels(labelIdx)
end

% Testing 32QAM test images
srcFiles = dir('testData\32QAM');
pwd = 'testData\32QAM';
for i = 3 : length(srcFiles)
filename = strcat('testData\32QAM\',srcFiles(i).name);
img = imread(fullfile(pwd,srcFiles(i).name));
[labelIdx, score] = predict(categoryClassifier,img);
srcFiles(i).name;
categoryClassifier.Labels(labelIdx)
end

% Testing 64QAM test images
srcFiles = dir('testData\64QAM');
pwd = 'testData\64QAM';
for i = 3 : length(srcFiles)
filename = strcat('testData\64QAM\',srcFiles(i).name);
img = imread(fullfile(pwd,srcFiles(i).name));
[labelIdx, score] = predict(categoryClassifier,img);
srcFiles(i).name;
categoryClassifier.Labels(labelIdx)
end