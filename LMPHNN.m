clc
clear

data = dlmread('数据\ai4i2020.txt');

labels = data(:, end);
num_classes = max(labels);
% 定义划分的比例（可以根据需要进行修改）
train_ratio = 0.5; % 训练集占总样本的比例
test_ratio = 0.5;  % 测试集占总样本的比例
% 删除在测试集中不存在的类别
unique_labels = unique(labels);
for i = 1:length(unique_labels)
    if nnz(labels == unique_labels(i)) == 1
        data(labels == unique_labels(i), :) = [];
        labels(labels == unique_labels(i)) = [];
    end
end
% 使用 cvpartition 创建随机划分
c = cvpartition(labels, 'HoldOut', test_ratio, 'Stratify', true);

% 获取训练集和测试集的逻辑索引
train_indices = training(c);
test_indices = test(c);

% 使用逻辑索引获取对应的训练集和测试集
train_data = data(train_indices, :);
train_labels = labels(train_indices);
test_data = data(test_indices, :);
test_labels = labels(test_indices);



result = zeros(0);

for z = 1:9
train_feature = train_data(:, 1:end-1);
train_label   = train_data(:, end);
test_feature  = test_data(:, 1:end-1);
test_label    = test_data(:, end);
KK         = max(test_label);
num_K      = 2:10;

%% find PNN

index_class              = zeros(0);
train_feature_each_class = zeros(0);
idx                      = zeros(0);
mean_value               = zeros(0);
dist                     = zeros(0);
distance                 = zeros(0);
for i = 1:KK
    index_class{i} = find(train_label == i);
    train_feature_each_class{i} = train_feature(index_class{i}, :);
    idx{i} = knnsearch(train_feature_each_class{i}, test_feature, 'dist', 'euclidean', 'k', num_K(z));
    
    %% compute local mean vector by LMKNN
%     for j = 1:size(test_feature, 1)
%         mean_value{i}(j, :) = mean(train_feature_each_class{i}(idx{i}(j, :), :));
%         dist(j, i) = pdist2(mean_value{i}(j, :), test_feature(j, :));
%     end
    
    %% compute local mean vector by LMPNN
    for j = 1:size(test_feature, 1)
        for k = 1:num_K(z)
            dists = pdist2(train_feature_each_class{i}(idx{i}(j, 1:k), :), test_feature(j, :), 'euclidean');
            mean_value{i, k}(j, :) = mean(train_feature_each_class{i}(idx{i}(j, 1:k), :), 1);
            dist{i}(j, k) = k / sum(1 ./ dists);
        end 
    end
    distance(:, i) = sum(dist{i}, 2);
end

  [~, I] = min(distance,[],2);

num_error = 0;
num_emptyset = 0;
num_metaclass = 0;
num_acc = 0;
num_tp = 0; % true positive
num_fp = 0; % false positive
num_fn = 0; % false negative
metaclass = zeros(0);
tt = 1;
for i = 1:length(I)
    if I(i) ~= test_label(i)
            num_error = num_error + 1;
            if I(i) == z % predicted as z, but actually not z
                num_fp = num_fp + 1;
            else % predicted as not z, but actually z
                num_fn = num_fn + 1;
            end
    elseif I(i) == test_label(i)
            num_acc = num_acc + 1;
    end
end

% calculate precision, recall, and F1 score
precision = num_acc / (num_acc + num_fp);
recall = num_acc / (num_acc + num_fn);
F1 = 2 * precision * recall / (precision + recall);

result(z, 1) = num_error / length(test_label);
% result(z, 2) = num_metaclass / length(test_label);
% result(z, 3) = num_emptyset / length(test_label);
result(z, 2) = (num_acc) / length(test_label);

result(z, 3) = recall;
result(z, 4) = precision;
result(z, 5) = F1;

end
Aresult = mean(result, 1);
% 

 