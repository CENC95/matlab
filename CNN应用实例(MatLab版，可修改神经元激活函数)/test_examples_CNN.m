%function [TrainingTime, TrainingAccuracy, TestingTime, TestingAccuracy] = test_examples_CNN(TrainingData, TestingData)
%{
clear all; close all; clc;
addpath('../data');
addpath('../util');
load mnist_uint8;
%}

%{
train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x  = double(reshape(test_x' , 28, 28, 10000)) / 255;
%}
%{
train_x = double(reshape(TrainingData(:, 2:end), 16, 16, size(TrainingData, 1)));
test_x  = double(reshape(TestingData( :, 2:end), 16, 16, size(TestingData , 1)));
train_y = double(TrainingData(:, 1));
test_y  = double(TestingData( :, 1));
%}
%%{
load 'Yale(28x28) 3-D.mat'
i = 1:165;
a = i(mod(i, 11) == 0);
b = i(~ismember(i, a));
train_x = Yale_Data(:, :, b);
test_x  = Yale_Data(:, :, a);
train_y = Yale_Label(b);
test_y  = Yale_Label(a);
%}
NumofLabel = size(unique(Yale_Label), 1);
train_y    = LabelPreprocess(train_y', NumofLabel);
test_y     = LabelPreprocess(test_y' , NumofLabel);
clear Yale_Data;
clear Yale_Label;

%% ex1 
%will run 1 e5poch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 20, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 20, 'kernelsize',5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    %}
};

% �����cnn�����ø�cnnsetup������ݴ˹���һ��������CNN���磬������
cnn = cnnsetup(cnn, train_x, train_y);

% ѧϰ��
opts.alpha = 0.055;%1;
% ÿ������һ��batchsize��batch��ѵ����Ҳ����ÿ��batchsize�������͵���һ��Ȩֵ��������
% �����������������ˣ�������������������˲ŵ���һ��Ȩֵ
opts.batchsize = 20; 
% ѵ����������ͬ��������������ѵ����ʱ��
% 1��ʱ�� 11.41% error
% 5��ʱ�� 4.2% error
% 10��ʱ�� 2.73% error
opts.numepochs = 5;

% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
start_time = cputime;
cnn = cnntrain(cnn, train_x, train_y, opts);
TrainingTime = cputime - start_time

a=1;
if a == 1
[er, bad] = cnntest(cnn, train_x, train_y);
TrainingAccuracy = 1 - er
end;

% Ȼ����ò�������������
start_time = cputime;
[er, bad] = cnntest(cnn, test_x, test_y);
TestingTime = cputime - start_time

TestingAccuracy = 1 - er
%{
%plot mean squared error
plot(cnn.rL);
%show test error
disp([num2str(er*100) '% error']);
%}