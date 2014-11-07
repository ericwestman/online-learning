% Kernelized SVM

%% load the data from files
clear all; close all; clc;

data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);


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


%% Run Kernelized SVM

% Initialize weights
w1 = zeros(size(f1,2),1); % initialize weights

% Iterate through training data
T = length(data1)/10;

% Weight the margin violation
lambda = 1;


for t = 1:T
    grad = lambda/T*w1;
    if -y1(t)*dot(w1,f1(t,:)) > -1 % violated constraint
        grad = grad - y1(t)*f1(t);
    end
    
    % Learning rate - make this better later
    alpha = 1/t;
    % Take gradient descent step
    w1 = w1 - alpha*grad;
    
end

% Predict on test data

% for t = 1:T
%     %test the algorithm
% end


