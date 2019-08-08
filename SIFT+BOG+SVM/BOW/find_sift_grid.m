function sift_arr = sp_find_sift_grid(I, grid_x, grid_y, patch_size, sigma_edge)

% sigma_edge 是正态分布的标准差，值越大，图像越模糊（平滑）

% parameters
num_angles = 8;  % 总共8个柱
num_bins = 4; 
num_samples = num_bins * num_bins;   % 1个区块划分为4*4区域
alpha = 9;  % 角度衰减参数(必须是奇数)

if nargin < 5  % nargin是用来判断输入变量的个数，是matlab的保留字
    sigma_edge = 1;
end

angle_step = 2 * pi / num_angles;   % 每45°一个柱
angles = 0:angle_step:2*pi;
angles(num_angles+1) = []; % bin centers  % 消除最后一个元素360°，使angles的大小为1*8，而不是1*9

[hgt wid] = size(I);
num_patches = numel(grid_x);  % number of   %576个patch

sift_arr = zeros(num_patches, num_samples * num_angles); %4*4*8=128  % 576*128维sift特征向量，初始化为0

[G_X,G_Y]=gen_dgauss(sigma_edge);  % 生成一个5*5的高斯模板（取值[-1,1]），并求出GX,GY的梯度G_X、G_Y

I_X = filter2(G_X, I, 'same'); % vertical edges   % 卷积，将G_X与图像I进行滤波叠加（注：图像I本身各点像素值就是归一化的数值[0,1]）
I_Y = filter2(G_Y, I, 'same'); % horizontal edges  % 同上，因为进行了滤波叠加，则此时图像中可能取值为[-1,1]
I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude  % 200*200的图像，计算图像梯度的幅值

I_theta = atan2(I_Y,I_X);   % 200*200的图像，图像梯度的幅角

I_theta(find(isnan(I_theta))) = 0; % necessary???  % 用0替代非法的结果，因为有些梯度方向为inf

% make default grid of samples (centered at zero, width 2)  % 确定图像采样点的默认（初始化）位置
interval = 2/num_bins:2/num_bins:2;
interval = interval - (1/num_bins + 1);  % [-0.75,-0.25,0.25,0.75]
[sample_x sample_y] = meshgrid(interval, interval);
sample_x = reshape(sample_x, [1 num_samples]); % change to array 1:16
sample_y = reshape(sample_y, [1 num_samples]);

% make orientation images   % 确定图像(200*200)，每一个像素点梯度的8个方向
I_orientation = zeros(hgt, wid, num_angles);  % z轴为8个方向
% for each histogram angle
for a=1:num_angles    
    % compute each orientation channel
    cos(I_theta - angles(a));
    tmp = cos(I_theta - angles(a)).^alpha;
    tmp = tmp .* (tmp > 0);
    
    % weight by magnitude
    I_orientation(:,:,a) = tmp .* I_mag;
end

% for all patches
for i=1:num_patches
    % 取第i个patch的坐标
    r = patch_size/2;  % r=8
    cx = grid_x(i) + r - 0.5;  
    cy = grid_y(i) + r - 0.5;

    % find coordinates of sample points (bin centers) % 查找16个采样点（16个bin中心）坐标
    sample_x_t = sample_x * r + cx;
    sample_y_t = sample_y * r + cy;
    sample_res = sample_y_t(2) - sample_y_t(1);  % 采样点间隔
    
    % find window of pixels that contributes to this descriptor  % 区块的上下界限
    x_lo = grid_x(i);
    x_hi = grid_x(i) + patch_size - 1;
    y_lo = grid_y(i);
    y_hi = grid_y(i) + patch_size - 1;
    
    % find coordinates of pixels    % 查找像素坐标
    [sample_px, sample_py] = meshgrid(x_lo:x_hi,y_lo:y_hi);
    num_pix = numel(sample_px);  % 256像素（因为区块大小是16*16）
    sample_px = reshape(sample_px, [num_pix 1]);  % 重构一个256行1列的数组
    sample_py = reshape(sample_py, [num_pix 1]);  % 重构一个256行1列的数组
        
    % find (horiz, vert) distance between each pixel and each grid sample
    % 一个区块256个像素到16个采样点（16个bin中心）的距离
    dist_px = abs(repmat(sample_px, [1 num_samples]) - repmat(sample_x_t, [num_pix 1]));  % 256*16
    dist_py = abs(repmat(sample_py, [1 num_samples]) - repmat(sample_y_t, [num_pix 1]));  % 256*16
    
    % find weight of contribution of each pixel to each bin  % 每个像素对16个采样点（16个bin中心）贡献的权重
    weights_x = dist_px/sample_res;  
    weights_x = (1 - weights_x) .* (weights_x <= 1);
    weights_y = dist_py/sample_res;
    weights_y = (1 - weights_y) .* (weights_y <= 1);
    weights = weights_x .* weights_y;  % 总权重为x，y方向权重的乘积
%     % make sure that the weights for each pixel sum to one?
%     tmp = sum(weights,2);
%     tmp = tmp + (tmp == 0);
%     weights = weights ./ repmat(tmp, [1 num_samples]);
        
    % make sift descriptor  % 构造sift特征
    curr_sift = zeros(num_angles, num_samples);  % 一个patch大小，8*16（8个方向，16个采样点（16个bin中心）
    for a = 1:num_angles 
        tmp = reshape(I_orientation(y_lo:y_hi,x_lo:x_hi,a),[num_pix 1]);  % 一个区块16*16共计256个像素点的第a个方向      
        tmp = repmat(tmp, [1 num_samples]); % 扩展到16个bin中心
        curr_sift(a,:) = sum(tmp .* weights); % 第a个方向要乘以该方向的权重weights
    end
    sift_arr(i,:) = reshape(curr_sift, [1 num_samples * num_angles]);  % 1*128（循环576次）
        
%     % visualization
%     if sigma_edge >= 3
%         subplot(1,2,1);
%         rescale_and_imshow(I(y_lo:y_hi,x_lo:x_hi) .* reshape(sum(weights,2), [y_hi-y_lo+1,x_hi-x_lo+1]));
%         subplot(1,2,2);
%         rescale_and_imshow(curr_sift);
%         pause;
%     end
end

function G=gen_gauss(sigma)  % 生成一个5*5的二维高斯滤波器（高斯模板是中心对称的）

if all(size(sigma)==[1, 1])
    % isotropic gaussian   % 各向同性高斯
	f_wid = 4 * ceil(sigma) + 1;  % 5
    G = fspecial('gaussian', f_wid, sigma);  % 生成一个5*5矩阵 σ=0.8
%	G = normpdf(-f_wid:f_wid,0,sigma);
%	G = G' * G;
else
    % anisotropic gaussian  % 各向异性高斯（横、纵坐标的σ取值不同）
    f_wid_x = 2 * ceil(sigma(1)) + 1;  
    f_wid_y = 2 * ceil(sigma(2)) + 1;
    G_x = normpdf(-f_wid_x:f_wid_x,0,sigma(1));
    G_y = normpdf(-f_wid_y:f_wid_y,0,sigma(2));
    G = G_y' * G_x;
end

function [GX,GY]=gen_dgauss(sigma)  % 得到delta高斯函数

% laplacian of size sigma
%f_wid = 4 * floor(sigma);
%G = normpdf(-f_wid:f_wid,0,sigma);
%G = G' * G;
G = gen_gauss(sigma);   % 生成一个二维高斯滤波器（5*5的高斯模板）

[GX,GY] = gradient(G);  % 计算G在X和Y方向的梯度GX、GY，得到delta高斯函数

GX = GX * 2 ./ sum(sum(abs(GX))); % colum sum and all sum
GY = GY * 2 ./ sum(sum(abs(GY)));

