% Multiclass SVM

%% load the data from files
clear all; close all; clc;

data_type = 'not dummy';
if strcmp(data_type, 'dummy')
    data1 = dlmread('data/dummyMulticlass1.txt','',3,0);
    data2 = dlmread('data/dummyMulticlass2.txt','',3,0);
    
    %Randomize the data
    data1=data1(randsample(1:length(data1),length(data1)),:);
    data2=data2(randsample(1:length(data2),length(data2)),:);
    
    % Class label definitions: class1, class2
    classes = [-1, 0, 1];
    
    % First dataset
    x1 = data1(:,1);        % x position
    y1 = data1(:,2);        % y position
    z1 = data1(:,3);        % z position
    i1 = data1(:,4);        % indices
    l1 = data1(:,5);        % labels
    f1 = data1(:,1:3);    % features

    % Second dataset
    x2 = data2(:,1);
    y2 = data2(:,2);
    z2 = data2(:,3);
    i2 = data2(:,4);
    l2 = data2(:,5);
    f2 = data2(:,1:3);     
    
else
    data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
    data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);
    
    %Randomize the data
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
    f2 = data2(:,6:end);    
end

%% Initialize multiclass SVM

% Define our training data
training_data = data1;
l_training = l1;
f_training = f1;
T1 = length(training_data);

% Define our test data
test_data = data1;
l_test = l1;
f_test = f1;
T2 = length(test_data);

if strcmp(data_type, 'dummy')
    % Define number of features / classes
    F = length(f_training);     % number of features
    C = 3;                      % number of classes
    labels1 = zeros(T1,C);
    labels2 = zeros(T2,C);
    
    for c = 1:C
        labels1(:,c) = ((l_training == classes(c)) - 0.5)*2;
        labels2(:,c) = ((l_test == classes(c)) - 0.5)*2;
    end
    
else
    % Define number of features / classes
    F = 10;     % number of features
    C = 5;      % number of classes
    labels1 = zeros(T1,C);
    labels2 = zeros(T2,C);

    for c = 1:C
        labels1(:,c) = ((l_training == classes(c)) - 0.5)*2;
        labels2(:,c) = ((l_test == classes(c)) - 0.5)*2;
    end
end

% Initialize weights for each class (each column is a different class)
w = zeros(size(f_training,2),C); % initialize weights

% Weight the margin violation
lambda = 10;

%% Run multiclass SVM
method = 'linear';
%% Linear - update every other class that is confusing
if strcmp(method,'linear')
    %Iterate this many times through the training data
    for iter = 1:1
        % Iterate through training data
        for t = 1:T1
            % Find which class this point belongs to
            c = find(labels1(t,:) == 1);
            other = find(labels1(t,:) == -1);

            %Iterate through all other classes
            for i = numel(other)
                nc = numel(i);
                
                % Calculate the normal shrinkage term of the gradient
                grad = lambda/T1*w(:,c);

                % Check if there's a violation with this class
                if (w(:,c)' - w(:,nc)')*f_training(t,:)' < 1 % violation
                    grad = grad - f_training(t,:)';
                end 

                % Learning rate - make this better later
                alpha = 1/((iter - 1)*T1 + t);

                % Take gradient descent step
                w(:,c) = w(:,c) - alpha*grad;
            end
        end
    end

%% Nonlinear - update only the most confusing class
else
    for iter = 1:1
        % Iterate through training data
        for t = 1:T1
            % Find which class this point belongs to
            c = find(labels1(t,:) == 1);
            nc = find(labels1(t,:) == -1);

            % Calculate the normal shrinkage term of the gradient
            grad = lambda/T1*w(:,c);

            % Find the most confusing class and see if there is a violation
            [~, mc] = max(w(:,nc)'*f_training(t,:)');
            if (w(:,c)' - w(:,mc)')*f_training(t,:)' < 1 % violation
                grad = grad - f_training(t,:)';
            end 

            % Learning rate - make this better later
            alpha = 1/((iter - 1)*T1 + t);

            % Take gradient descent step
            w(:,c) = w(:,c) - alpha*grad;
        end
    end
end
%% Predict the test data
% Define the correct labels for the test data
pred_label = zeros(T2,1);
error = zeros(C,1);

% Predict on test data
for t = 1:T2
    % Find which class this point belongs to and record it
    class = find(labels2(t,:) == 1);
    nc = find(labels2(t,:) == -1);

    [~, pred_label(t)] = max(w(:,1:C)'*f_test(t,:)');
    
    if pred_label(t) ~= class
        error(class) = error(class) + 1;
    end
end

total_classes = sum(labels2 == 1);

% Display the error of the test data
total_error = sum(error)/T2;
error = error ./ total_classes';
error
total_error

%% Write data to a PCD file
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
'VIEWPOINT 400 400 30 1 0 0 0\n',...
strcat(sprintf('POINTS %d \n',T2),'\n'),...
'DATA ascii\n'));

for i = 1:T2
%     fprintf(pcd, '%f %f %f\n', x1(i), y1(i), z1(i));

    % Set the color for each class
    if(pred_label(i) == 1)
        color = 4.808e+06;
    else
        color = 4.2108e+06;
    end
    
    fprintf(pcd, '%f %f %f %f \n', x1(i), y1(i), z1(i), color);
end

fclose(pcd);




























