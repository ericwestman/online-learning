% binary SVM

%% load the data from files
clear all; close all; clc;

data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

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

for c = 1:C
    labels1(:,c) = ((l1 == classes(c)) - 0.5)*2;
    labels2(:,c) = ((l2 == classes(c)) - 0.5)*2;
end

% Select which class we're going to look at
% Here, 4 = ground
label = labels1(:,4);

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
label = labels2(:,4);
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

% Write data to a PCD file
pcd = fopen('kernelizedSVM1.pcd','w');

if (pcd < 0)
    error('Could not open kernelizedSVM1.pcd');
end

fprintf(pcd,strcat('# .PCD v.7 - Point Cloud Data file format\n',...
'VERSION .7\n',...
'FIELDS x y z rgb\n',... % 'FIELDS x y z rgb\n',...
'SIZE 4 4 4 4\n',... % 'SIZE 4 4 4 4\n',...
'TYPE F F F F\n',...
'COUNT 1 1 1 1\n',...
strcat(sprintf('WIDTH %d',T2),'\n'),...
'HEIGHT 1\n',...
'VIEWPOINT 0 0 0 1 0 0 0\n',...
strcat(sprintf('POINTS %d \n',T2),'\n'),...
'DATA ascii\n'));

for i = 1:T2
%     fprintf(pcd, '%f %f %f\n', x1(i), y1(i), z1(i));

    % Set the color for each class
    if(pred_lab(i) == 1)
        color = 4.808e+06;
    else
        color = 4.2108e+06;
    end
    
    fprintf(pcd, '%f %f %f %f \n', x2(i), y2(i), z2(i), color);
end

fclose(pcd);




























