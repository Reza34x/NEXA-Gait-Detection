 %% Import Datas(with my labels 0)
clear
load("datas.mat")
columns_to_remove = [1, 20:25, 34, 36, 38:39, 41, 44, 46, 47, 49, 50];
datas(:, columns_to_remove) = [];




%% Find peaks(my labels1)

fsr = datas(:,28); %max = HS, min = TO

[High_pks,High_locs] = findpeaks(fsr, "MinPeakProminence", 0.45, 'MinPeakHeight',0.08); %storing the value and location of high peaks
[Low_pks,Low_locs] = findpeaks(-(fsr), "MinPeakProminence",0.25); %storing the value and location of low peaks

figure()
plot(fsr, LineWidth=1.5)
hold on
plot(High_locs-10, High_pks,  "Marker","v", "MarkerSize", 7, "LineStyle","none", Color='red')
hold on
plot(Low_locs, -Low_pks,  "Marker","^", "MarkerSize", 7, "LineStyle", "none", Color='magenta')
legend("data", "High peaks", "Low peaks")
title("Peak detection")
%% make labels(My labels1)
n = 1;
m = 1;
label = zeros(length(datas), 1);
for i=1:length(datas)
    if n<=57 && i == High_locs(n)
        n= n+1;
        label(i) = 1;
    elseif m<=53 && i == Low_locs(m)
        m=m+1;
        label(i) = 2;
    else 
        label(i) = 0;
    end
    
end

labels = categorical(label);

%% Find peaks(my labels2)

rows_to_remove = [1:1446, 4581:7311, 10305:14858, 17794:23939, 26952:32329, 35288:38390, 41423:44890];
datas(rows_to_remove, :) = []; %Remove no walking data
fsr = datas(:,45); %min = heel off


[High_pks,High_locs] = findpeaks(fsr, "MinPeakProminence", 0.001); %storing the value and location of high peaks

for i = 1:length(High_locs)
    if i<= length(High_locs)

        if High_pks(i)>0.1
            High_pks(i) = NaN;
            High_locs(i) = NaN;
            i = i - 1;
        end
    end
end
High_pks = High_pks(~isnan(High_pks));
High_locs = High_locs(~isnan(High_locs));
clear i 

figure()
plot(datas(:, 45), LineWidth=1)
hold on
plot(High_locs, High_pks,  "Marker","^", "MarkerSize", 7, "LineStyle", "none", Color='magenta')
legend("data", "High peaks")
title("Peak detection")
%% make labels(My labels2)
% High_locs = High_locs-70;

labels = datas(:, end); %labeling

columns_to_remove = [1, 20:25, 34, 36, 38:39, 41, 44, 46, 47, 49, 50:53]; % Remove NaN data
datas(:, columns_to_remove) = [];

%Delete label 2(Mid Stance)
for i = 1:length(labels)
    if labels(i, end) == 2       
        labels(i, end) = 0;
    end
end


% add label 2(Heel Off) from algorithm
% n = 1;
% for i=1:length(datas)
%     if n<=length(High_pks) && i == High_locs(n)
%         n= n+1;
%         for j = -10:10
%             labels(i + j) = 2;
%         end
%     else 
%         label(i) = 0;
%     end
% 
% end

% add label 2(Heel Off) from Manual
n = 1;
for i=1:length(datas)
    if n<=length(my_locs) && i == my_locs(n)
        n= n+1;
        for j = -5:5
            labels(i + j) = 2;
        end
    else 
        label(i) = 0;
    end
    
end

labels = categorical(labels);

clear i j n columns_to_remove rows_to_remove




%% --------------------------------------------------------------------------------------------
% Delete label 1+3

% for i = 1:length(datas)
%     if datas(i, end) == 1 
%         datas(i, end) = 1;
%     elseif datas(i, end) == 3 
%         datas(i, end) = 2;
%     elseif datas(i, end) == 2 
%         datas(i, end) = 0;
%     end
% end
% clear i
%-------------------------------------------------------------------
%% Import Datas(containing actual labels)
clear
load("datas.mat")
rows_to_remove = [1:1446, 4581:7311, 10305:14858, 17794:23939, 26952:32329, 35288:38390, 41423:44890];
datas(rows_to_remove, :) = []; %Remove no walking data

labels = categorical(datas(:, end)); %labeling

columns_to_remove = [1, 20:25, 34, 36, 38:39, 41, 44, 46, 47, 49, 50:53]; % Remove NaN data
datas(:, columns_to_remove) = [];

% datas = datas(:, [26, 32]);


% %% PCA
% [coeff, score, latent, ~] = pca(datas);
% totalVar = sum(latent);
% cumulativeVar = cumsum(latent) / totalVar;
% numPCs = find(cumulativeVar > 0.95, 1);
% 
% % Project the data onto the principal components
% X_pca = score(:, 1:numPCs);
% 
% % Visualize the explained variance ratio
% figure;
% plot(1:length(latent), cumulativeVar);
% xlabel('Number of principal components');
% ylabel('Cumulative explained variance ratio');
% title('Scree plot');
% 
% % Find the largest coefficient values for each principal component
% top_coeffs = coeff(:, 1:end);
% 
% [~, max_idx] = max(abs(top_coeffs), [], 1);
% disp("Top features are : ")
% disp(max_idx)
% %% Drop columns base of PCA
% 
% datas = datas(:, unique(max_idx(1:20))); %keep top 10


%% training/validation/test
numtraining = round(length(datas) * 0.7);
numvalidation = round(length(datas) * 0.15);
numtest = length(datas)-numvalidation-numtraining;
clear training_input validation_input test_input training_label validation_label test_label

for i = 1:numtraining
    training_input{i,:} = datas(i,:)';
    training_label(i,1) = labels(i,1);
end

for i = 1:numvalidation
    validation_input{i,:} = datas(i+numtraining, :)';
    validation_label(i,1) = labels(i+numtraining,1);
end

for i = 1:numtest
    test_input{i,:} = datas(i+numtraining+numvalidation,:)';
    test_label(i,1) = labels(i+numtraining+numvalidation,1);
end

clear i

XTrain = padsequences(training_input,2);
XTest = padsequences(test_input,2);
XValid = padsequences(validation_input,2);


adsXTrain = arrayDatastore(XTrain,'IterationDimension',3);
adsYTrain = arrayDatastore(training_label);

adsXTest = arrayDatastore(XTest,'IterationDimension',3);
adsYTest = arrayDatastore(test_label);

adsXValid = arrayDatastore(XValid,'IterationDimension',3);
adsYValid = arrayDatastore(validation_label);

cdsTrain = combine(adsXTrain,adsYTrain);
cdsValid = combine(adsXValid,adsYValid);

%% Network 1: Only Dense layers



opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.001,...
    "MaxEpochs",10,...
    "MiniBatchSize",128,...
    "ValidationData",cdsValid,...
    "Verbose",true,...
    "Shuffle","every-epoch",...
    'Verbose', 0, ...
    'OutputNetwork','best-validation-loss', ...
    "Plots","training-progress");

layers1 = [
    sequenceInputLayer(size(datas, 2),"Name","sequence")
    fullyConnectedLayer(128,"Name","fc")
    reluLayer("Name","relu")
    batchNormalizationLayer()
    fullyConnectedLayer(64,"Name","fc2")
    reluLayer("Name","relu2")
    batchNormalizationLayer()
    fullyConnectedLayer(4,"Name","fc_1")
    softmaxLayer("Name","softmax")
    classificationLayer('Name', 'classoutput', 'Classes', ...
    categorical([0, 1, 2, 3]), 'ClassWeights', [0.150, 1.2500, 1.2500, 1.2500])
     ];

% analyzeNetwork(layers1);
% plot(layers);
% lgraph = layerGraph(layers1);
% plot(lgraph)
[net1, traininfo] = trainNetwork(cdsTrain,layers1,opts);


%%  test the network 1

predict_label_cell = classify(net1,adsXTest);
predict_label = categorical;

num_of_nonzero_labels = length(find(test_label ~= "0"));

for i = 1:length(predict_label_cell)
    
    predict_label(i,1) =  predict_label_cell{i, 1};
    
end

accuarcy = 100 * mean( predict_label == test_label);
disp("the accuracy of network is " +  accuarcy + "%")

disp("the number of misclassified points is " + ...
    length(find(predict_label ~= test_label)) + " of " + num_of_nonzero_labels);

% 
C = confusionmat(test_label, predict_label);

% Calculate precision for each class
numClasses = size(C, 1);
precision = zeros(numClasses, 1);

for i = 1:numClasses
    tp = C(i, i); % True positives
    fp = sum(C(:, i)) - tp; % False positives
    precision(i) = tp / (tp + fp);
end

% Calculate overall precision (macro-average)
overall_precision = mean(precision);
disp("the precision of the model is: "+overall_precision)
% 
plotconfusion(test_label, predict_label);

plot(predict_label, LineWidth=1)
hold on
plot(test_label, LineWidth=1)
title('Dense Layer');

legend({"predicted", "actual"})
%% Class weight (just to show)

classWeights = 1./countcats(training_label);
classWeights = [0.0005, 0.0025, 0.0025, 0.0025];
classWeights = classWeights'/mean(classWeights);

%% Network 2: CNN

opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.1,...
    "MaxEpochs",10,...
    "MiniBatchSize",128,...
    "ValidationData",cdsValid,...
    "Verbose",true,...
    "Shuffle","every-epoch",...
    'Verbose', 0, ...
    'OutputNetwork','best-validation-loss', ...
    "Plots","training-progress");

layers2 = [
    sequenceInputLayer(33,"Name","sequence")
    convolution1dLayer(3,8,"Name","conv1d","Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(3,"Name","maxpool1d","Padding","same")
    flattenLayer("Name","flatten")
    lstmLayer(8)
    fullyConnectedLayer(16,"Name","fc")
    reluLayer
    batchNormalizationLayer
    fullyConnectedLayer(4,"Name","fc1")
    softmaxLayer("Name","softmax")
    classificationLayer('Name', 'classoutput', 'Classes', ...
    categorical([0, 1, 2, 3]), 'ClassWeights', [0.030, 1.2500, 1.3000, 1.2500])
];

[net2, traininfo] = trainNetwork(cdsTrain,layers2,opts);

%%  test the network 2

predict_label_cell = classify(net2,adsXTest);
predict_label = categorical;

num_of_nonzero_labels = length(find(test_label ~= "0"));

for i = 1:length(predict_label_cell)
    
    predict_label(i,1) =  predict_label_cell{i, 1};
    
end

accuarcy = 100 * mean( predict_label == test_label);
disp("the accuracy of network is " +  accuarcy + "%")

disp("the number of misclassified points is " + ...
    length(find(predict_label ~= test_label)) + " of " + num_of_nonzero_labels);
% 
C = confusionmat(test_label, predict_label);

% Calculate precision for each class
numClasses = size(C, 1);
precision = zeros(numClasses, 1);

for i = 1:numClasses
    tp = C(i, i); % True positives
    fp = sum(C(:, i)) - tp; % False positives
    precision(i) = tp / (tp + fp);
end

% Calculate overall precision (macro-average)
overall_precision = mean(precision);
disp("the precision of the model is: "+overall_precision)
% 
plotconfusion(test_label, predict_label);

plot(test_label, LineWidth=1)
hold on
plot(predict_label, LineWidth=1)
title('CNN');


legend({"actual", "predicted"})

%% Network 3 : LSTM

opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",10,...
    "MiniBatchSize",64,...
    "ValidationData",cdsValid,...
    "Verbose",true,...
    "Shuffle","every-epoch",...
    'Verbose', 0, ...
    'OutputNetwork','best-validation-loss', ...
    "Plots","training-progress");

layers3 = [
    sequenceInputLayer(33,"Name","sequence")
    flattenLayer("Name","flatten")
    lstmLayer(128, "OutputMode","sequence")
    fullyConnectedLayer(1024,"Name","fc")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer('Name', 'classoutput', 'Classes', ...
    categorical([0, 1, 2, 3]), 'ClassWeights', [0.1500, 1.2500, 1.5000, 1.2500])
];

[net3, traininfo] = trainNetwork(cdsTrain,layers3,opts);

%%  test the network 3

predict_label_cell = classify(net3,adsXTest);
predict_label = categorical;

num_of_nonzero_labels = length(find(test_label ~= "0"));

for i = 1:length(predict_label_cell)
    
    predict_label(i,1) =  predict_label_cell{i, 1};
    
end

accuarcy = 100 * mean( predict_label == test_label);
disp("the accuracy of network is " +  accuarcy + "%")

disp("the number of misclassified points is " + ...
    length(find(predict_label ~= test_label)) + " of " + num_of_nonzero_labels);

% 
C = confusionmat(test_label, predict_label);

% Calculate precision for each class
numClasses = size(C, 1);
precision = zeros(numClasses, 1);

for i = 1:numClasses
    tp = C(i, i); % True positives
    fp = sum(C(:, i)) - tp; % False positives
    precision(i) = tp / (tp + fp);
end

% Calculate overall precision (macro-average)
overall_precision = mean(precision);
disp("the precision of the model is: "+overall_precision)
% 

plotconfusion(test_label, predict_label);

plot(test_label, LineWidth=1)
hold on
plot(predict_label, LineWidth=1)
title('LSTM');


legend({"actual", "predicted"})

%% Network 4 : LSTM + custom

opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.0001,...
    "MaxEpochs",10,...
    "MiniBatchSize",32,...
    "ValidationData",cdsValid,...
    "Verbose",true,...
    "Shuffle","every-epoch",...
    'Verbose', 0, ...
    'OutputNetwork','best-validation-loss', ...
    "Plots","training-progress");

layers4 = [
    sequenceInputLayer(33,"Name","sequence")
    flattenLayer("Name","flatten")
    lstmLayer(128, "OutputMode","sequence")
    fullyConnectedLayer(128,"Name","fc1")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(64,"Name","fc2")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(4,"Name","output")
    softmaxLayer("Name","softmax")
    classificationLayer('Name', 'classoutput', 'Classes', ...
    categorical([0, 1, 2, 3]), 'ClassWeights', [0.1500, 1.2500, 1.3000, 1.2500])
];

[net4, traininfo] = trainNetwork(cdsTrain,layers4,opts);
% [0.03, 1.2500, 1.2500, 1.2500]

%%  test the network 4

predict_label_cell = classify(net4,adsXTest);
predict_label = categorical;

num_of_nonzero_labels = length(find(test_label ~= "0"));

for i = 1:length(predict_label_cell)
    
    predict_label(i,1) =  predict_label_cell{i, 1};
    
end

accuarcy = 100 * mean( predict_label == test_label);
disp("the accuracy of network is " +  accuarcy + "%")

disp("the number of misclassified points is " + ...
    length(find(predict_label ~= test_label)) + " of " + num_of_nonzero_labels);

plotconfusion(test_label, predict_label);

plot(test_label, LineWidth=1)
hold on
plot(predict_label, LineWidth=1)
title('Custom LSTM');


legend({"actual", "predicted"})

%% Network 5: Attention + LSTM

opts = trainingOptions("adam",...
    "ExecutionEnvironment","gpu",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",10,...
    "MiniBatchSize",64,...
    "ValidationData",cdsValid,...
    "Verbose",true,...
    "Shuffle","every-epoch",...
    'Verbose', 0, ...
    'OutputNetwork','best-validation-loss', ...
    "Plots","training-progress");

layers5 = [
    sequenceInputLayer(33,"Name","sequence")
    selfAttentionLayer(64, 128, "Name","selfattention")
    layerNormalizationLayer("Name","layernorm")
    lstmLayer(32,"Name","lstm")
    fullyConnectedLayer(1024,"Name","fc")
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)
    fullyConnectedLayer(4,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer('Name', 'classoutput', 'Classes', ...
    categorical([0, 1, 2, 3]), 'ClassWeights', [0.1500, 1.2500, 1.3000, 1.2500])
];
% [0.150, 1.3, 1.2500]

[net5, traininfo] = trainNetwork(cdsTrain,layers5,opts);

%%  test the network 5

predict_label_cell = classify(net5,adsXTest);
predict_label = categorical;

num_of_nonzero_labels = length(find(test_label ~= "0"));

for i = 1:length(predict_label_cell)
    
    predict_label(i,1) =  predict_label_cell{i, 1};
    
end

accuarcy = 100 * mean( predict_label == test_label);
disp("the accuracy of network is " +  accuarcy + "%")

disp("the number of misclassified points is " + ...
    length(find(predict_label ~= test_label)) + " of " + num_of_nonzero_labels);
% 
C = confusionmat(test_label, predict_label);

% Calculate precision for each class
numClasses = size(C, 1);
precision = zeros(numClasses, 1);

for i = 1:numClasses
    tp = C(i, i); % True positives
    fp = sum(C(:, i)) - tp; % False positives
    precision(i) = tp / (tp + fp);
end

% Calculate overall precision (macro-average)
overall_precision = mean(precision);
disp("the precision of the model is: "+overall_precision)
% 

plotconfusion(test_label, predict_label);

plot(test_label, LineWidth=1)
hold on
plot(predict_label, LineWidth=1)
title('Attention + LSTM');


legend({"actual", "predicted"})




