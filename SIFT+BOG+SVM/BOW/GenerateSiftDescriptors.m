function [] = GenerateSiftDescriptors(opts,descriptor_opts)

fprintf('Building Sift Descriptors\n\n');

%% parameters
descriptor_flag=1;
maxImageSize = descriptor_opts.maxImageSize;
gridSpacing = descriptor_opts.gridSpacing;
patchSize = descriptor_opts.patchSize;

try
    descriptor_opts2=getfield(load([opts.globaldatapath,'/',descriptor_opts.name,'_settings']),'descriptor_opts');
    
    if(isequal(descriptor_opts,descriptor_opts2))
        descriptor_flag=0;
        display('descriptor has already been computed for this settings');
    else
        display('Overwriting descriptor with same name, but other descriptor settings !!!!!!!!!!');
    end
end

if(descriptor_flag)   % 如果sift特征没有被计算
    
    %% load image
    load(opts.image_names);         % load image in data set
    nimages=opts.nimages;           % number of images in data set
    
    for f = 1:nimages
        
        I=load_image([opts.imgpath,'/', image_names{f}]); % 调用函数，将每一张图片变为灰度图（像素取值归一化[0,1]）
        
        [hgt wid] = size(I);
        if min(hgt,wid) > maxImageSize  % 图片大小预处理
            I = imresize(I, maxImageSize/min(hgt,wid), 'bicubic');
            fprintf('Loaded %s: original size %d x %d, resizing to %d x %d\n', ...
                image_names{f}, wid, hgt, size(I,2), size(I,1));
            [hgt wid] = size(I);
        end
        
       %% make grid (coordinates of upper left patch corners)
       
        remX = mod(wid-patchSize,gridSpacing);% the right edge
        offsetX = floor(remX/2)+1;
        remY = mod(hgt-patchSize,gridSpacing);
        offsetY = floor(remY/2)+1;
        
        %生成图片划分的网格 24*24=576个patch
        [gridX,gridY] = meshgrid(offsetX:gridSpacing:wid-patchSize+1, offsetY:gridSpacing:hgt-patchSize+1);
  
        fprintf('Processing %s: wid %d, hgt %d, grid size: %d x %d, %d patches\n', ...
            image_names{f}, wid, hgt, size(gridX,2), size(gridX,1), numel(gridX));
        
       %% find SIFT descriptors
        siftArr = find_sift_grid(I, gridX, gridY, patchSize, 0.8); % 计算576个区块的 Dense Sift特征
        siftArr = normalize_sift(siftArr);  % 归一化
        
        features.data = siftArr;
        features.x = gridX(:) + patchSize/2 - 0.5;  % 保存patch中心
        features.y = gridY(:) + patchSize/2 - 0.5;
        features.wid = wid;
        features.hgt = hgt;
        features.patchSize=patchSize;
        
        image_dir=sprintf('%s/%s/',opts.localdatapath,num2string(f,3)); % location descriptor
    
        mkdir(opts.localdatapath,num2string(f,3));
        save ([image_dir,'/','sift_features'], 'features');           % save the descriptors
        
        fprintf('The %d th image finished...\n',f);
       
    end % for
    save ([opts.globaldatapath,'/',descriptor_opts.name,'_settings'],'descriptor_opts');      % save the settings of descriptor in opts.globaldatapath
end % if

end% function
