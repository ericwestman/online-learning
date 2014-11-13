% Multiclass SVM

%% load the data from files
clear all; close all; clc;

data_type = 'not dummy';
if strcmp(data_type, 'dummy')
    data1 = dlmread('data/dummyMulticlass1.txt','',3,0);
    data2 = dlmread('data/dummyMulticlass2.txt','',3,0);
    
    % Randomize the data
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
    f1 = data1(:,1:3);      % features

    % Second dataset
    x2 = data2(:,1);
    y2 = data2(:,2);
    z2 = data2(:,3);
    i2 = data2(:,4);
    l2 = data2(:,5);
    f2 = data2(:,1:3);     
    
    % Normalize features
%     for i = 1:numel(x1)
%         f1(i,:) = f1(i,:)./norm(f1(i,:));
%     end
%     
%     for i = 1:numel(x2)
%         f2(i,:) = f2(i,:)./norm(f2(i,:));
%     end
    
else
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
    f2 = data2(:,6:end);    
    
    % Normalize features
%     for i = 1:numel(x1)
%         f1(i,:) = f1(i,:)./norm(f1(i,:));
%     end
%     
%     for i = 1:numel(x2)
%         f2(i,:) = f2(i,:)./norm(f2(i,:));
%     end
    
end

% Define our training data
training_data = data1;
l_training = l1;
f_training = f1;
T1 = length(training_data);

% Define our test data
test_data = data2;
l_test = l2;
f_test = f2;
T2 = length(test_data);


%% Initialize multiclass SVM

if strcmp(data_type, 'dummy')
    % Define number of features / classes
    F = length(f_training(1,:));    % number of features
    C = length(classes);                          % number of classes
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

% Set up kernel and function
% RBF kernel 
sigma = 1;
kernel = @(x,x1) exp(-1*(norm(x - x1).^2)./(2*sigma^2));

% Regularization weight
lambda = 0.3;
% Max number of kernels
k_max = 100;

for i = 1:C
    alphas{i} = zeros(k_max,1);     % Kernel weights
    xi{i} = zeros(k_max,F);   	% Kernel locations
end;


% Weight the margin violation
lambda = 0.01;

%% Run multiclass SVM
tic
%Iterate this many times through the training data
for iter = 1:1
    % Iterate through training data
    for t = 1:T1/10
        % Find which class this point belongs to
        c = find(labels1(t,:) == 1);
        other = find(labels1(t,:) == -1);
        
        learning_rate = 0.3;

        %Iterate through all other classes
        for i = 1:numel(other)
            nc = other(i);
                        
            % Shrink all the weights to begin with
            alphas{c} = (1-2*lambda*learning_rate)*alphas{c};

            % Check if there's a violation with this class
            if (func(kernel, alphas{c}, xi{c}, f_training(t,:)) - ...
                    func(kernel, alphas{nc}, xi{nc}, f_training(t,:)) ) < 1 % violation
                
                t
                % Add a kernel at this feature vector in the correct class
                [~, c_ind] = min(abs(alphas{c}));
                
                alphas{c}(c_ind) = learning_rate;
                xi{c}(c_ind,1:F) = f_training(t,:);
                
                % Add a kernel at this feature vector in the wrong class
                [~, nc_ind] = min(abs(alphas{c}));
                alphas{nc}(nc_ind) = -learning_rate;
                xi{nc}(nc_ind,1:F) = f_training(t,:);
                
            end
        end
    end
end
toc
%% Predict the test data
% Define the correct labels for the test data
pred_label = zeros(T2,1);
error = zeros(C,1);
tic
% Predict on test data
for t = 1:T2
    % Find which class this point belongs to and record it
    class = find(labels2(t,:) == 1);
    nc = find(labels2(t,:) == -1);

    for c = 1:C
        weights(c) = func(kernel, alphas{c}, xi{c}, f2(t,:));
    end
    
    [~, pred_label(t)] = max(weights);
    
    if pred_label(t) ~= class
        error(class) = error(class) + 1;
    end
end
toc
total_classes = sum(labels2 == 1);

% Display the error of the test data
total_error = sum(error)/T2;
error = error ./ total_classes';
error
total_error

%% Write data to a PCD file
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
% 'VIEWPOINT 400 400 30 1 0 0 0\n',...
% strcat(sprintf('POINTS %d \n',T2),'\n'),...
% 'DATA ascii\n'));
% 
% for i = 1:T2
% %     fprintf(pcd, '%f %f %f\n', x1(i), y1(i), z1(i));
% 
%     % Set the color for each class
%     if(pred_label(i) == 1)
%         color = 4.808e+06;
%     else
%         color = 4.2108e+06;
%     end
%     
%     fprintf(pcd, '%f %f %f %f \n', x1(i), y1(i), z1(i), color);
% end
% 
% fclose(pcd);

































