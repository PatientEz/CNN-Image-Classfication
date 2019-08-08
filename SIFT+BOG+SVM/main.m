%% Script to perform BOW-based image classification demo
% ========================================================================
% Image Classification using Bag of Words and Spatial Pyramid BoW
% Created by Piji Li (pagelee.sd@gmail.com)  
% Blog: 丕子 http://www.zhizhihu.com
% QQ: 379115886
% IRLab. : http://ir.sdu.edu.cn     
% Shandong University,Jinan,China
% 10/24/2011
%% initialize the settings
disp('*********** start *********')
clc;
ini;
detect_opts=[];descriptor_opts=[];dictionary_opts=[];assignment_opts=[];ada_opts=[];

%% Descriptors
descriptor_opts.type='sift';                                                     % name descripto
descriptor_opts.name=['des',descriptor_opts.type]; % output name (combines detector and descrtiptor name)
descriptor_opts.patchSize=16;                                                   % normalized patch size
descriptor_opts.gridSpacing=8;   % 每个区块的步长
descriptor_opts.maxImageSize=1000;
GenerateSiftDescriptors(pg_opts,descriptor_opts);  % 产生每张图片的576*128维的sift特征，共计300张图片

%% Create the texton dictionary
dictionary_opts.dictionarySize = 500;
dictionary_opts.name='sift_features';
dictionary_opts.type='sift_dictionary';
CalculateDictionary(pg_opts, dictionary_opts);  % K-Means对240张训练集照片的总共240*576*128行进行聚类，得到300*128个聚类中心

%% assignment
assignment_opts.type='1nn';                                 % name of assignment method  % 1NN算法
assignment_opts.descriptor_name=descriptor_opts.name;       % name of descriptor (input)
assignment_opts.dictionary_name=dictionary_opts.name;       % name of dictionary
assignment_opts.name=['BOW_',descriptor_opts.type];         % name of assignment output
assignment_opts.dictionary_type=dictionary_opts.type;
assignment_opts.featuretype=dictionary_opts.name;
assignment_opts.texton_name='texton_ind';
do_assignment(pg_opts,assignment_opts);  % 构造BOW（300*360），每一列代表一张图像的频率分布直方图（归一化，每一列和为1）
% 该词包模型BOW存储在BOW_sift.mat文件下，需要用load加载BOW_sift.mat，才可以在Workspace显示词袋模型BOW

%% CompilePyramid
% 作者提出的另外一种统计词频的方法，用来构造词袋模型，分为3层，分别三次，每次将图片分为4*4,2*2,1*1，共计21块
% 对每一块进行300个聚类中心的聚类，故得到的词袋pyramid_all大小为（21*300）* 360 块，总计6300类*360
% 该词袋模型存储在spatial_pyramid.mat文件下，需要用load加载spatial_pyramid.mat，才可以在Workspace显示pyramid_all
pyramid_opts.name='spatial_pyramid';
pyramid_opts.dictionarySize=dictionary_opts.dictionarySize;
pyramid_opts.pyramidLevels=3;  % 金字塔层数
pyramid_opts.texton_name=assignment_opts.texton_name;
CompilePyramid(pg_opts,pyramid_opts); % 构造另外一个词袋模型（6300*360），每一列代表一张图像的频率分布直方图（归一化，每一列和为1）

%% Classification
do_classification_rbf_svm  % 利用BOW+径向基核函数rbf的SVM进行分类

%% histogram intersection kernel
do_classification_inter_svm  % 利用BOW+直方图交叉核（作者定义的）SVM进行分类

%% pyramid bow rbf
do_p_classification__rbf_svm  % 利用pyramid_all+径向基核函数rbf的SVM进行分类

%% pyramid bow histogram intersection kernel
do_p_classification__inter_svm  % 利用pyramid_all+直方图交叉核（作者定义的）SVM进行分类 % 最高精度 ≈90.83%

%%
do_classification_liner_svm   % 利用BOW+线性核函数的SVM进行分类

%%
do_p_classification__liner_svm   % 利用pyramid_all+线性核函数的SVM进行分类

%
show_results_script  % 用最高精度绘图（混淆矩阵）（横着看）

%% AdaBoost  
% 采用AdaBoost进行分类
ada_opts.T = 100;
ada_opts.weaklearner  = 0;
ada_opts.epsi = 0.2;
ada_opts.lambda = 1e-3;
ada_opts.max_ite = 3000;
ada_opts.bow = assignment_opts.name;
ada_opts.pbow = pyramid_opts.name;
% do_classification_adaboost_bow(pg_opts,ada_opts);
% do_classification_adaboost_pyramid_bow(pg_opts,ada_opts);

