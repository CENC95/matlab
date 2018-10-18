
%{
load("Yale(28x28) 4-D");

i = 1:165;
a = i(mod(i, 11) == 0);
b = i(~ismember(i, a));
train_x = Yale_Data(:, :, :, b);
test_x  = Yale_Data(:, :, :, a);
train_y = categorical(Yale_Label(b));
test_y  = categorical(Yale_Label(a));
%}

%%{
load('YaleB(192x168) 3-D');

train_x = ThreeD2FourD(TrainingData.img );
test_x  = ThreeD2FourD(TestingData.img  );
train_y = categorical(TrainingData.label);
test_y  = categorical(TestingData.label );
%}

%%{
%建立自己的网络
layers = [imageInputLayer([192 168 1], 'DataAugmentation', 'randfliplr');%();
          convolution2dLayer(5,20);%, 'WeightL2Factor',30, 'BiasL2Factor', 30);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(5,20);%, 'WeightL2Factor', 30, 'BiasL2Factor', 30);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          %{
          convolution2dLayer(4,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          %}
          dropoutLayer(0.5);
          fullyConnectedLayer(numel(categories([train_y; test_y])));
          softmaxLayer;
          classificationLayer()];
      
%设定训练参数
options = trainingOptions('sgdm','MaxEpochs',500,...
    'InitialLearnRate',0.0001,...
    'L2Regularization', 0.05);
%}

%训练网络
convnet = trainNetwork(train_x,train_y,layers,options);
%测试网络
YTrain = classify(convnet, train_x);
TTrain = train_y;
YTest  = classify(convnet,test_x);
TTest  = test_y;
TrainingAccuracy = sum(YTrain == TTrain) / numel(TTrain)
TestingAccuracy  = sum(YTest  == TTest ) / numel(TTest)
