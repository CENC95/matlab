
clear all; 
close all; clc;
load mnist_uint8;

train_x = double(reshape(train_x', 28, 28, 60000)) / 255;
test_x  = double(reshape(test_x' , 28, 28, 10000)) / 255;

%% ex1 
%will run 1 e5poch in about 200 second and get around 11% error. 
%With 100 epochs you'll get around 1.2% error

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps',12, 'kernelsize',5) %convolution layer
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
opts.numepochs = 500;

% Ȼ��ʼ��ѵ��������������ʼѵ�����CNN����
start_time = cputime;
cnn = cnntrain(cnn, train_x, train_y, opts);
TrainingTime = cputime - start_time

a=1;
if a == 1
[er, bad] = cnntest(cnn, train_x, train_y);
TrainingAccuracy = 1 - er
end

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