%% show results
% shows the most probable images for each of the classes

load(pg_opts.trainset);
load(pg_opts.testset);
load(pg_opts.labels);

indexes=1:pg_opts.nimages;  % 1*360
test_indexes=indexes(testset);  % 1*120

num_class=pg_opts.nclasses;  % 6
num_test_1c=floor(size(predict_label,1)/num_class);  % 20

confusion_matrix=size(num_class,num_class);  % 初始化混淆矩阵的大小

for ci=1:num_class
    for cj=1:num_class
       confusion_matrix(ci,cj)=size(find(predict_label((ci-1)*num_test_1c+1:ci*num_test_1c,:)==cj),1)/num_test_1c; % 生成混淆矩阵
    end
end
draw_cm(confusion_matrix,pg_opts.classes,num_class);
