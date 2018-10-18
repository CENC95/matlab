%function [TrainingTime, TrainingAccuracy, TestingTime, TestingAccuracy] = BP(TrainingData, TestingData, hiddenSizes)

% 功能： 
%     使用BP神经网络对数据进行分类识别，作为结果，给出训练和测试的准确率及所需时长
%
% 输入：
%     TrainingData - 以MATLAB的矩阵变量形式存储的训练用数据（矩阵每行表示一个带标签的数据样本（向量），矩阵第一列为数据的标签，第二列至末列为组成一个数据样本的各个元素）
%     TestingData  - 以MATLAB的矩阵变量形式存储的测试用数据（数据格式与TrainingData相同）
%     hiddenSizes  - 指定神经网络的隐含层数和各隐含层神经元数，其数据结构为至少1个元素的整型数组，数组元素的个数表示隐含层数，每个元素的值表示对应隐含
%                   层的神经元数，如：
%                       hiddenSizes的值为[10]，则该BP神经网络为单隐含层、隐含层神经元数为10的网络结构；
%                       hiddenSizes的值为[10,20]，则该BP神经网络为双隐含层，两个隐含层神经元数分别为10、20的网络结构
%                   
%
% 输出：
%     TrainingTime     - 训练时间
%     TrainingAccuracy - 训练准确率
%     TestingTime      - 测试时间
%     TestingAccuracy  - 测试准确率

load('forTest_BP')
hiddenSizes = 10;


%读取训练数据
train_data_class = TrainingData(:, 1)';
train_data       = TrainingData(:, 2:size(TrainingData, 2))';
clear TrainingData;

%读取测试数据
test_data_class  = TestingData( :, 1)';
test_data        = TestingData( :, 2:size(TestingData, 2))'; 
clear TestingData;

NumberofTrainingData = size(train_data_class,2);
NumberofTestingData  = size(test_data_class ,2);

%标签预处理（与elm算法中对标签的预处理步骤相同）
sorted_target = sort(cat(2, train_data_class, test_data_class), 2);
label         = zeros(1,1);                               
label(1,1)    = sorted_target(1,1);

j=1;
for i = 2:(NumberofTrainingData + NumberofTestingData)
    if sorted_target(1,i) ~= label(1,j)
        j = j + 1;                              
        label(1,j) = sorted_target(1,i);
    end
end
number_class = j;

NumberofOutputNeurons = number_class;
temp=zeros(NumberofOutputNeurons, NumberofTrainingData);
for i = 1:NumberofTrainingData
    for j = 1:number_class
        if label(1,j) == train_data_class(1,i)
            break; 
        end
    end
    temp(j,i)=1;
end
train_data_class = temp * 2 - 1;

temp=zeros(NumberofOutputNeurons, NumberofTestingData);
for i = 1:NumberofTestingData
    for j = 1:number_class
        if label(1,j) == test_data_class(1,i)
            break; 
        end
    end
    temp(j,i)=1;
end
test_data_class = temp * 2 - 1;

%特征值归一化
leg        = sqrt( sum(train_data.^2) );
train_data = bsxfun(@rdivide, train_data, leg);%train_data=train_data/leg;
leg        = sqrt( sum(test_data.^2) );
test_data  = bsxfun(@rdivide, test_data, leg);

%创建神经网络，‘traingdx’表示使用梯度下降自适应学习率训练函数
net = feedforwardnet(hiddenSizes, 'traingdx');

%将网络的迭代次数设置为100
net.trainparam.epochs = 100 ;

%记录训练开始时间
start_time_train = cputime;

%开始训练
net = train( net, train_data , train_data_class) ;

%计算训练时间
end_time_train = cputime;

TrainingTime   = end_time_train-start_time_train

Y = sim(net, train_data);

%记录测试开始时间
start_time_test = cputime;

%仿真
TY = sim(net , test_data) ;

%计算测试时间
end_time_test = cputime;
TestingTime   = end_time_test-start_time_test   

%统计识别正确率
MissClassification_Train = 0;
MissClassification_Test  = 0;

[~, expectType] = max(Y);
[~, actulaType] = max(train_data_class);
for i = 1:size(Y, 2)
    if expectType(i) ~= actulaType(i)
        MissClassification_Train = MissClassification_Train + 1;
    end
end

TrainingAccuracy = 1 - MissClassification_Train / size(Y, 2)

[~, expectType] = max(TY);
[~, actulaType] = max(test_data_class);
for i = 1 : size(TY, 2)
    if expectType(i) ~= actulaType(i)
        MissClassification_Test = MissClassification_Test + 1 ; 
    end
end

TestingAccuracy = 1 - MissClassification_Test / size(TY, 2)
