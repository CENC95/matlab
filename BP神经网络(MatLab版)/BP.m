%function [TrainingTime, TrainingAccuracy, TestingTime, TestingAccuracy] = BP(TrainingData, TestingData, hiddenSizes)

% ���ܣ� 
%     ʹ��BP����������ݽ��з���ʶ����Ϊ���������ѵ���Ͳ��Ե�׼ȷ�ʼ�����ʱ��
%
% ���룺
%     TrainingData - ��MATLAB�ľ��������ʽ�洢��ѵ�������ݣ�����ÿ�б�ʾһ������ǩ�������������������������һ��Ϊ���ݵı�ǩ���ڶ�����ĩ��Ϊ���һ�����������ĸ���Ԫ�أ�
%     TestingData  - ��MATLAB�ľ��������ʽ�洢�Ĳ��������ݣ����ݸ�ʽ��TrainingData��ͬ��
%     hiddenSizes  - ָ������������������͸���������Ԫ���������ݽṹΪ����1��Ԫ�ص��������飬����Ԫ�صĸ�����ʾ����������ÿ��Ԫ�ص�ֵ��ʾ��Ӧ����
%                   �����Ԫ�����磺
%                       hiddenSizes��ֵΪ[10]�����BP������Ϊ�������㡢��������Ԫ��Ϊ10������ṹ��
%                       hiddenSizes��ֵΪ[10,20]�����BP������Ϊ˫�����㣬������������Ԫ���ֱ�Ϊ10��20������ṹ
%                   
%
% �����
%     TrainingTime     - ѵ��ʱ��
%     TrainingAccuracy - ѵ��׼ȷ��
%     TestingTime      - ����ʱ��
%     TestingAccuracy  - ����׼ȷ��

load('forTest_BP')
hiddenSizes = 10;


%��ȡѵ������
train_data_class = TrainingData(:, 1)';
train_data       = TrainingData(:, 2:size(TrainingData, 2))';
clear TrainingData;

%��ȡ��������
test_data_class  = TestingData( :, 1)';
test_data        = TestingData( :, 2:size(TestingData, 2))'; 
clear TestingData;

NumberofTrainingData = size(train_data_class,2);
NumberofTestingData  = size(test_data_class ,2);

%��ǩԤ������elm�㷨�жԱ�ǩ��Ԥ��������ͬ��
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

%����ֵ��һ��
leg        = sqrt( sum(train_data.^2) );
train_data = bsxfun(@rdivide, train_data, leg);%train_data=train_data/leg;
leg        = sqrt( sum(test_data.^2) );
test_data  = bsxfun(@rdivide, test_data, leg);

%���������磬��traingdx����ʾʹ���ݶ��½�����Ӧѧϰ��ѵ������
net = feedforwardnet(hiddenSizes, 'traingdx');

%������ĵ�����������Ϊ100
net.trainparam.epochs = 100 ;

%��¼ѵ����ʼʱ��
start_time_train = cputime;

%��ʼѵ��
net = train( net, train_data , train_data_class) ;

%����ѵ��ʱ��
end_time_train = cputime;

TrainingTime   = end_time_train-start_time_train

Y = sim(net, train_data);

%��¼���Կ�ʼʱ��
start_time_test = cputime;

%����
TY = sim(net , test_data) ;

%�������ʱ��
end_time_test = cputime;
TestingTime   = end_time_test-start_time_test   

%ͳ��ʶ����ȷ��
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
