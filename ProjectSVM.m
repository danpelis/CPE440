pwd = 'C:\Users\dppel\Projects\CPE 440';

trainingSetPath = fullfile(pwd,'trainingData');
testSetPath = fullfile(pwd,'testData');

imds = imageDatastore(trainingSetPath,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(testSetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');

bag = bagOfFeatures(trainingSet);

categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);

confMatrix = evaluate(categoryClassifier,testSet)

mean(diag(confMatrix))

img = imread(fullfile(pwd,'\testData\16QAM\','16QAM_245.jpg'));
[labelIdx, score] = predict(categoryClassifier,img);

categoryClassifier.Labels(labelIdx)