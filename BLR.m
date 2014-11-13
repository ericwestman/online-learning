% Multiclass SVM

%% load the data from files
clear all; close all;

% Data1 will be our training data
data2 = dlmread('data/oakland_part3_am_rf.node_features','',3,0);
% Data2 will be our test data
data1 = dlmread('data/oakland_part3_an_rf.node_features','',3,0);

% Randomize the data
data1=data1(randsample(1:length(data1),length(data1)),:);
data2=data2(randsample(1:length(data2),length(data2)),:);

% Class label definitions: veg, wire, pole, ground, facade
classes = [1004, 1100, 1103, 1200, 1400];
selected_classes = [1, 2];

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
F = length(f1(1,:));
C = length(classes);
labels1 = zeros(T1,C);
labels2 = zeros(T2,C);

for c = 1:C
    labels1(:,c) = ((l1 == classes(c)) - 0.5)*2;
    labels2(:,c) = ((l2 == classes(c)) - 0.5)*2;
end

%% do BLR

% Define our training data
training_data = data1;
T1 = length(training_data);

% Define our test data
test_data = data2;
T2 = length(test_data);

% Select which class we're going to look at
% Here, 4 = ground
label = labels1(:,4);

% Initialize weights 
w1 = zeros(F,1); % initialize weights
J = zeros(F,1);

P = inv(10*eye(F));
sigma = 2;

% Train the data

for i = 1:T1
    P = P + 1/sigma^2*f1(i,:)*f1(i,:)';
    J = J + 1/sigma^2*f1(i,:);
end

w = inv(P)*j;

% Test the data






























