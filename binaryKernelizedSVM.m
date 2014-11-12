% binary Kernelized SVM

%% load the data from files
clear all; close all; clc;

data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

% Randomize the data
data1=data1(randsample(1:length(data1),length(data1)),:);
data2=data2(randsample(1:length(data2),length(data2)),:);

% Class label definitions: veg, wire, pole, ground, facade
classes = [1004, 1100, 1103, 1200, 1400];

% First dataset
x1 = data1(:,1);        % x position
y1 = data1(:,2);        % y position
z1 = data1(:,3);        % z position
i1 = data1(:,4);        % indices
l1 = data1(:,5);        % labels
f1 = data1(:,6:end);    % features

% Second dataset
x2 = data2(:,1);
y2 = data2(:,2);
z2 = data2(:,3);
i2 = data2(:,4);
l2 = data2(:,5);
f2 = data2(:,6:end);    % features


%% Run binary SVM

% Define our training data
training_data = data1;
T1 = length(training_data);

% Define our test data
test_data = data2;
T2 = length(test_data);

% Define number of features / classes
F = 10;     % number of features
C = 5;      % number of classes
labels1 = zeros(T1,C);
labels2 = zeros(T2,C);

% Generate -1 / 1 labels
for c = 1:C
    labels1(:,c) = ((l1 == classes(c)) - 0.5)*2;
    labels2(:,c) = ((l2 == classes(c)) - 0.5)*2;
end

num_classes1 = sum((labels1+1)/2);
num_classes2 = sum((labels2+1)/2);

% Select which class we're going to look at
% Here, 4 = ground
label = labels1(:,4);

% Set up kernel and function
% RBF kernel 
sigma = 5;
kernel = @(x,x1) exp(-1*(norm(x - x1).^2)./(2*sigma^2));

% Regularization weight
lambda = 0.3;
k_max = 500;
nk = 0;

alphas = [];    % Kernel weights
xi = [];         % Kernel locations

tic
%Iterate this many times through the training data
for iter = 1:1
    % Iterate through training data
    for t = 1:T1/10
        learning_rate = 1/10;
        
        % Shrink all the weights to begin with
        alphas = (1-2*lambda*learning_rate)*alphas;
        
        % Check if margin violation
        if (1 - label(t)*func(kernel, alphas, xi, f1(t,:)) ) > 0 % violation
            t
            % Remember that we're adding a kernel
            nk = nk + 1;
            
            % Add a kernel at this feature vector
            ind = mod(nk-1,k_max)+1;
            alphas(ind) = learning_rate*label(t);
            xi(ind,1:F) = f1(t,:);
            
        end

    end
end
toc

% Define the correct labels for the test data
test_label = labels2(:,4);
pred_lab = zeros();
error = 0;

tic
% Predict on test data
for t = 1:T2
    pred_lab(t) = sign(func(kernel, alphas, xi, f2(t,:)));
    if pred_lab(t) ~= test_label(t)
        error = error + 1;
    end
end
toc

% Display the error of the test data
error = error / T2

% Write data to a PCD file
% pcd = fopen('kernelizedSVM1.pcd','w');
% 
% if (pcd < 0)
%     error('Could not open kernelizedSVM1.pcd');
% end
% 
% fprintf(pcd,strcat('# .PCD v.7 - Point Cloud Data file format\n',...
% 'VERSION .7\n',...
% 'FIELDS x y z rgb\n',... % 'FIELDS x y z rgb\n',...
% 'SIZE 4 4 4 4\n',... % 'SIZE 4 4 4 4\n',...
% 'TYPE F F F F\n',...
% 'COUNT 1 1 1 1\n',...
% strcat(sprintf('WIDTH %d',T2),'\n'),...
% 'HEIGHT 1\n',...
% 'VIEWPOINT 0 0 0 1 0 0 0\n',...
% strcat(sprintf('POINTS %d \n',T2),'\n'),...
% 'DATA ascii\n'));
% 
% for i = 1:T2
% %     fprintf(pcd, '%f %f %f\n', x1(i), y1(i), z1(i));
% 
%     % Set the color for each class
%     if(pred_lab(i) == 1)
%         color = 4.808e+06;
%     else
%         color = 4.2108e+06;
%     end
%     
%     fprintf(pcd, '%f %f %f %f \n', x2(i), y2(i), z2(i), color);
% end
% 
% fclose(pcd);




























