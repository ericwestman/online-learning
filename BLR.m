% Multiclass SVM

%% load the data from files
clear all; close all;

% Data1 will be our training data
data1 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
% Data2 will be our test data
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

% Define number of features / classes
T1 = length(l1);
T2 = length(l2);
F = length(f2(1,:));

%% Set up training data and test data
l_training = [];
f_training = [];
x_test = [];
y_test = [];
z_test = [];
l_test = [];
f_test = [];

% Class label definitions: veg, wire, pole, ground, facade
% classes = [1004, 1100, 1103, 1200, 1400];

% class1 = 1004;
% class2 = 1200;

% class1 = 1100;
% class2 = 1103;

% class1 = 1400;
% class2 = 1103;

class1 = 1004;
class2 = 1100;

class1 = 1004;
class2 = 1400;



for i = 1:T1
    if (l1(i) == class1)
        l_training(end+1) = 1;
        f_training(end+1,:) = f1(i,:);
    elseif (l1(i) == class2)
        l_training(end+1) = -1;
        f_training(end+1,:) = f1(i,:);
    end
end

for i = 1:T2
    if (l2(i) == class1)
        l_test(end+1) = 1;
        x_test(end+1,:) = x2(i,:);
        y_test(end+1,:) = y2(i,:);
        z_test(end+1,:) = z2(i,:);
        f_test(end+1,:) = f2(i,:);
    elseif (l2(i) == class2)
        x_test(end+1,:) = x2(i,:);
        y_test(end+1,:) = y2(i,:);
        z_test(end+1,:) = z2(i,:);
        l_test(end+1) = -1;
        f_test(end+1,:) = f2(i,:);
    end
end

% Define our training data
TR = length(f_training);

% Define our test data
TE = length(f_test);

%% Standardize the features
f_training = f_training - repmat(mean(f_training),TR,1);
f_training = f_training ./ repmat(std(f_training),TR,1);

for i = 1:TR
    if (isnan(f_training(i,end)))
        f_training(i,end) = 1;
    end
end

f_test = f_test - repmat(mean(f_test),TE,1);
f_test = f_test ./ repmat(std(f_test),TE,1);

for i = 1:TE
    if (isnan(f_test(i,end)))
        f_test(i,end) = 1;
    end
end

%% do BLR


% Initialize Js
J = zeros(F,1);

P = inv(100*eye(F));
sigma = 1;

% Train the data
for i = 1:TR
    P = P + 1/sigma^2*f_training(i,:)*f_training(i,:)';
    J = J + 1/sigma^2*l_training(i)*f_training(i,:)';
end

w = inv(P)*J;

%% Test the data
pred_label = zeros(TE,1);
error = zeros(2,1);
labeled_test = zeros(TE,1);
% Predict on test data
for t = 1:TE
    pred_label(t) = sign(dot(w,f_test(t,:)));
    
    if pred_label(t) ~= l_test(t)
        if l_test(t) == 1
            error(1) = error(1) + 1;
        else
            error(2) = error(2) + 1;
        end
    end
end
total_classes(1) = sum(l_test == 1);
total_classes(2) = sum(l_test == -1);

% Display the error of the test data
total_error = sum(error)/TE;
error = error ./ total_classes';
error

%% Write data to a file of the same format as the original
file = fopen('data1BLR.pcd','w');

if (file < 0)
    error('Could not open file');
end

fprintf(file,strcat('#\n#\n#\n'));

for i = 1:TE
    % Get the right class
    if pred_label(i) == 1
        class = class1;
    else
        class = class2;
    end
    fprintf(file, '%f %f %f %d %d \n', x_test(i), y_test(i), z_test(i), i, class);
end

fclose(file);
























