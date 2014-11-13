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

% First dataset
x1 = data1(:,1);        % x position
y1 = data1(:,2);        % y position
z1 = data1(:,3);        % z position
i1 = data1(:,4);        % indices
l1 = data1(:,5);        % labels
f1 = data1(:,6:end-1);    % features

% Second dataset
x2 = data2(:,1);
y2 = data2(:,2);
z2 = data2(:,3);
i2 = data2(:,4);
l2 = data2(:,5);
f2 = data2(:,6:end-1);

% Normalize features
%     for i = 1:numel(x1)
%         f1(i,:) = f1(i,:)./norm(f1(i,:));
%     end
%     
%     for i = 1:numel(x2)
%         f2(i,:) = f2(i,:)./norm(f2(i,:));
%     end

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

%% Initialize kernelized SVM

% Only consider an equal number of data points from each class
num_points1 = sum((labels1+1)/2);
num_points2 = sum((labels2+1)/2);

% Define training data
% c_max = min(num_points1);
% counts = zeros(1,C);
% 
% TR = c_max*C;
% 
% l_training = zeros(c_max*C,C);
% f_training = zeros(c_max*C,F);
% 
% i = 1;
% while (sum(counts) < c_max*C )
%     this_class = find(labels1(i,:) == 1);
%     if counts(this_class) < c_max
%         counts = counts + (labels1(i,:) == 1);
%         l_training(sum(counts),:) = labels1(i,:);
%         f_training(sum(counts),:) = f1(i,:);
%     end
%     i = i+1;
% end
% 
% rand_ind = randsample(1:TR,TR);
% 
% l_training = l_training(rand_ind,:);
% f_training = f_training(rand_ind,:);

% Or copy points from less populated classes
copies = floor(max(num_points1)./num_points1);
TR = copies*num_points1';
indices = zeros(TR,1);
count = 1;
for i = 1:T1
    this_class = find(labels1(i,:) == 1);
    indices(count:count+copies(this_class)) = i;
    count = count + copies(this_class);
end

rand_ind = randsample(1:TR,TR);
repopulated_ind = indices(rand_ind);

l_training = labels1(repopulated_ind,:);
f_training = f1(repopulated_ind,:);
TR = TR / 2;

% Or just ignore that and pass in all the data
% l_training = labels1;
% f_training = f1;
% TR = T1;

% Define our test data
x_test = x2;
y_test = y2;
z_test = z2;
test_data = data2;
l_test = labels2;
f_test = f2;
TE = length(l_test);


% TR = TR / 10;
% TE = TE / 10;

%% Parameters for kernelized SVM

% Set up kernel and function
% RBF kernel 
sigma = 1;
kernel = @(x,x1) exp(-1*(norm(x - x1).^2)./(2*sigma^2));

% Regularization weight
lambda = 0.000001;
% Max number of kernels
k_max = 70;

for i = 1:C
    alphas{i} = zeros(k_max,1);     % Kernel weights
    xi{i} = zeros(k_max,F);   	% Kernel locations
end;

%% Run kernelized SVM
tic
% Iterate this many times through the training data
for iter = 1:1
    % Iterate through training data
    positive_examples = zeros(1,C);
    negative_examples = zeros(1,C);
    for t = 1:TR
        % Find which class this point belongs to
        c = find(l_training(t,:) == 1);
        other = find(l_training(t,:) == -1);
        
        learning_rate = 1/sqrt(TR);

        % Iterate through all other classes
        for i = 1:numel(other)
            nc = other(i);
            
            % Check to if we've already seen enough examples of this
            if negative_examples(nc) > num_points1(nc)
%                 continue;
            end

            % Check if there's a violation with this class
            if (func(kernel, alphas{c}, xi{c}, f_training(t,:)) - ...
                    func(kernel, alphas{nc}, xi{nc}, f_training(t,:)) ) < 1 % violation
                
                % Add a kernel at this feature vector in the correct class
                [~, c_ind] = min(abs(alphas{c}));
                alphas{c}(c_ind) = learning_rate;
                xi{c}(c_ind,1:F) = f_training(t,:);
                
                % Increment count of positive examples
                positive_examples(c) = positive_examples(c) + 1;
                
                % Add a kernel at this feature vector in the wrong class
                [~, nc_ind] = min(abs(alphas{c}));
                alphas{nc}(nc_ind) = -learning_rate;
                xi{nc}(nc_ind,1:F) = f_training(t,:);
                
                % Increment count of negative examples
                negative_examples(nc) = negative_examples(nc) + 1;
            else
                % pass
            end
        end
        
        % Shrink all the weights
        alphas{c} = (1-2*lambda*learning_rate)*alphas{c};
    
    end
end
toc
%% Predict the test data
% Define the correct labels for the test data
pred_label = zeros(T2,1);
error = zeros(C,1);
tic
% Predict on test data
for t = 1:TE
    % Find which class this point belongs to and record it
    class = find(l_test(t,:) == 1);
    nc = find(l_test(t,:) == -1);

    for c = 1:C
        weights(c) = func(kernel, alphas{c}, xi{c}, f_test(t,:));
    end
    
    [~, pred_label(t)] = max(weights);
    
    if pred_label(t) ~= class
        error(class) = error(class) + 1;
    end
end
toc
total_classes = sum(labels2(1:TE,:) == 1);

% Display the error of the test data
total_error = sum(error)/TE;
error = error ./ total_classes';
lambda
learning_rate
error
total_error

%% Write data to a file of the same format as the original
file = fopen('data1KSVM.pcd','w');

if (file < 0)
    error('Could not open file');
end

fprintf(file,strcat('#\n#\n#\n'));

for i = 1:TE
    % Set the color for each class
    fprintf(file, '%f %f %f %d %d \n', x_test(i), y_test(i), z_test(i), i, classes(pred_label(i)));
end

fclose(file);

































