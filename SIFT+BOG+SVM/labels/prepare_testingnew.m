clc; clear;
% load image_names;
% load labels;
% load testset;
% load trainset;
clc; clear;
load image_names;
load labels;
load testset;
load trainset;
%之前有2236个  trian
 preimagelength = 2236;
 maindir = 'E:\CV作业\MyWorkSpace\images\testing';
 subdir =  dir( maindir );   % 先确定子文件夹
 disp(length(subdir))
for i = 1 : length( subdir )
    if( isequal( subdir( i ).name, '.' ) || ...
        isequal( subdir( i ).name, '..' ) || ...
        ~subdir( i ).isdir )   % 如果不是目录跳过
        continue;
    end
    %从i = 3开始
    subdirpath = fullfile( maindir, subdir( i ).name, '*.jpg' );
    images = dir( subdirpath );   % 在这个子文件夹下找后缀为jpg的文件
    disp(i)
    disp(subdirpath)
    disp(length(images))
    % 遍历每张图片
    for j = 1 : length( images )
        l = preimagelength + j;
        imagepath = fullfile( subdir( i ).name, images( j ).name );
        image_names{l}=['testing\',imagepath];
        labels(l,1)=i-2;
        trainset(l,1)=0;
        testset(l,1)=1;  
               
    end
    preimagelength = preimagelength +length(images)
end
 

%一共4433个   图片    2236 train 2197 test

save('image_names','image_names');
save('labels','labels');
trainset=logical(trainset);
testset=logical(testset);
save('trainset','trainset');
save('testset','testset');







