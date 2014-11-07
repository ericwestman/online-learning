% Kernelized SVM

%% load the data from files
clear all; close all; clc;

data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
data2 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

% Class label definitions: veg, wire, pole ground, facade
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


%% Run Kernelized SVM

% Define our training data
training_data = data1;
T = length(training_data);

% Define number of features / classes
F = 10;     % number of features
C = 5;      % number of classes
labels1 = zeros(T,C);
for c = 1:C
    labels1(:,c) = (l1 == classes(c) - 0.5)*2;
end

% Initialize weights for each class (each column is a different class)
w1 = zeros(size(f1,2),C); % initialize weights

% Weight the margin violation
lambda = 10;

%Iterate this many times through the training data!
for runs = 1:1
    % Iterate through training data
    for t = 1:T
        %Iterate through each class
        for c = 1:C
            % Calculate the normal shrinkage term of the gradient
            grad = lambda/T*w1(:,c);

            % Check if constraint is violated (need margin)
            if -labels1(t)*dot(w1(:,c),f1(t,:)) > -1 % violated constraint
                grad = grad - labels1(t)*f1(t);
            end 

            % Learning rate - make this better later
            alpha = 1/t;

            % Take gradient descent step
            w1(:,c) = w1(:,c) - alpha*grad;
        end
    end
end
% Predict on test data

% for t = 1:T
%     %test the algorithm
% end


% Write data to a PCD file
pcd = fopen('kernelizedSVM1.pcd','w');

if (pcd < 0)
    error('Could not open kernelizedSVM1.pcd');
end

fprintf(pcd,strcat('# .PCD v.7 - Point Cloud Data file format\n',...
'VERSION .7\n',...
'FIELDS x y z\n',... % 'FIELDS x y z rgb\n',...
'SIZE 4 4 4\n',... % 'SIZE 4 4 4 4\n',...
'TYPE F F F\n',...
'COUNT 1 1 1\n',...
strcat(sprintf('WIDTH %d',length(x1)),'\n'),...
'HEIGHT 1\n',...
'VIEWPOINT 400 400 30 1 0 0 0\n',...
strcat(sprintf('POINTS %d \n',length(x1)),'\n'),...
'DATA ascii\n'));

for i = 1:length(x1)
    fprintf(pcd, '%f %f %f\n', x1(i), y1(i), z1(i));
    %fprintf(pcd, '%d %d %d\n', x1(i), y1(i), z1(i), dot(w1,f1(t,:)));
end

fclose(pcd);




























