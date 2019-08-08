% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (peegeelee@gmail.com)  
% Blog: http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% classification script using SVM

fprintf('\nClassification using BOW linear_svm\n');
% load the BOW representations, the labels, and the train and test set
load(pg_opts.trainset);
load(pg_opts.testset);
load(pg_opts.labels);
load([pg_opts.globaldatapath,'/',assignment_opts.name]) % 从BOW_sift中加载BOW词袋模型(300*360)

train_labels    = labels(trainset);          % contains the labels of the trainset % A(B)代表A的第i个元素是A(B(i))，但是如果B中有0，那么B一定要是logical矩阵，不然报错
train_data      = BOW(:,trainset)';          % contains the train data
[train_labels,sindex]=sort(train_labels);    % we sort the labels to ensure that the first label is '1', the second '2' etc
train_data=train_data(sindex,:);
test_labels     = labels(testset);           % contains the labels of the testset
test_data       = BOW(:,testset)';           % contains the test data

%% here you should of course use crossvalidation !

%%
bestc=200;bestg=2;
bestcv=0;
% for log2c = -1:10,
%   for log2g = -1:0.1:1.5,
%     cmd = ['-v 5 -t 2 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
%     cv = svmtrain(train_labels, train_data, cmd);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end

%% 分类
% -s 0 多类分类 C-SVC
% -t 2 内核函数 rbf
% -c 200 惩罚因子
% -b 1 概率估计：是否计算SVC或SVR的概率估计，可选值0 或1，默认0； （如果需要估计分到每个类的概率 b=1，则需要设置这个）
% -g 2 核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)(默认1/ k)
options=sprintf('-s 0 -t 0 -c %f -b 1 -g %f -q',bestc,bestg);
model = svmtrain(train_labels, train_data,options);

%% 预测
% predict_label : 120*1，记录每张照片所位于的类别
% accuracy : 3*1，包括正确率，均方误差，相关系数
% dec_values :  120*6，每行是预测的结果（概率分布），概率大的即为分类结果 predict_label
[predict_label, accuracy , dec_values] = svmpredict(test_labels,test_data, model,'-b 1');
