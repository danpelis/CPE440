%{
-----------------------------------------
--Prepare Images by making them 126x126--
-----------------------------------------
%}

srcFiles = dir('C:\Users\dppel\Projects\CPE 440\trainingData\16QAM');

for i = 3 : length(srcFiles)
filename = strcat('C:\Users\dppel\Projects\CPE 440\trainingData\16QAM\',srcFiles(i).name);
im = imread(filename);
k=imresize(im,[126,126]);
newfilename=strcat('C:\Users\dppel\Projects\CPE 440\trainingData\16QAM\',srcFiles(i).name);
imwrite(k,newfilename,'png');
end

srcFiles = dir('C:\Users\dppel\Projects\CPE 440\trainingData\32QAM');

for i = 3 : length(srcFiles)
filename = strcat('C:\Users\dppel\Projects\CPE 440\trainingData\32QAM\',srcFiles(i).name);
im = imread(filename);
k=imresize(im,[126,126]);
newfilename=strcat('C:\Users\dppel\Projects\CPE 440\trainingData\32QAM\',srcFiles(i).name);
imwrite(k,newfilename,'png');
end

srcFiles = dir('C:\Users\dppel\Projects\CPE 440\trainingData\64QAM');

for i = 3 : length(srcFiles)
filename = strcat('C:\Users\dppel\Projects\CPE 440\trainingData\64QAM\',srcFiles(i).name);
im = imread(filename);
k=imresize(im,[126,126]);
newfilename=strcat('C:\Users\dppel\Projects\CPE 440\trainingData\64QAM\',srcFiles(i).name);
imwrite(k,newfilename,'png');
end


%{
------------------------------------------
--Train the CNN with the prepared Images--
------------------------------------------
%}
digitDatasetPath = fullfile(pwd,'trainingData');


imds = imageDatastore(digitDatasetPath, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


numTrainingFiles = 300;


[imdsTrain, imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

%{
-----------------------------------------
------Can output the current labels------
-----------------------------------------
%}

tbl = countEachLabel(imds);

layers = [ imageInputLayer([126 126 3])
convolution2dLayer(5,20)
reluLayer
maxPooling2dLayer(2,'Stride',2)
fullyConnectedLayer(3)
softmaxLayer
classificationLayer];

options = trainingOptions('sgdm','MaxEpochs',20,'InitialLearnRate',1e-4,'Verbose',false, 'Plots','training-progress');
net = trainNetwork(imds,layers,options);


%{
-----------------------------------------
--------Test exmaple inputs in net-------
-----------------------------------------
%}

filename = strcat('C:\Users\dppel\Projects\CPE 440\testData\64QAM_187.jpg');
im = imread(filename);
k=imresize(im,[126,126]);
newfilename=strcat('C:\Users\dppel\Projects\CPE 440\testData\64QAM_187.jpg');
imwrite(k,newfilename,'png');

digitDatasetPath1 = fullfile(pwd,'testData\64QAM_187.jpg');
imds1 = imageDatastore(digitDatasetPath1, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
Ypred = classify(net,imds1);
imshow('testData\64QAM_187.jpg');
title(string(Ypred(1)));
Ypred;