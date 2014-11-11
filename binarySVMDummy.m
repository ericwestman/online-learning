% binary SVM

%% load the data from files
clear all; close all; clc;

data1 = dlmread('data/dummyData1.txt','',3,0);
data2 = dlmread('data/dummyData2.txt','',3,0);

% First dataset
x1 = data1(:,1);        % x position
y1 = data1(:,2);        % y position
z1 = data1(:,3);        % z position
i1 = data1(:,4);        % indices
l1 = data1(:,5);        % labels
f1 = data1(:,1:3);      % features

% First dataset
x2 = data2(:,1);        % x position
y2 = data2(:,2);        % y position
z2 = data2(:,3);        % z position
i2 = data2(:,4);        % indices
l2 = data2(:,5);        % labels
f2 = data2(:,1:3);      % features

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

% Select which class we're going to look at
% Here, 4 = ground
label = l1;

% Initialize weights for each class (each column is a different class)
w1 = zeros(size(f1,2),1); % initialize weights

% Weight the margin violation
lambda = 1;

%Iterate this many times through the training data
for iter = 1:1
    % Iterate through training data
    for t = 1:T1
        % Calculate the normal shrinkage term of the gradient
        grad = lambda/T1*w1;

        % Check if constraint is violated (need margin)
        if label(t)*dot(w1,f1(t,:)) < 1 % violation
            grad = grad - label(t)*f1(t,:)';
        end 

        % Learning rate - make this better later
        alpha = 1/((iter - 1)*T1 + t);

        % Take gradient descent step
        w1 = w1 - alpha*grad;
    end
end

% Define the correct labels for the test data
label = l2;
pred_lab = zeros();
error = 0;

% Predict on test data
for t = 1:T2
    pred_lab(t) = sign(dot(w1,f2(t,:)));
    if pred_lab(t) ~= label(t)
        error = error + 1;
    end
end

% Display the error of the test data
error = error / T2





























