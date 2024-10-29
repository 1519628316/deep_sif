clear
train = 0;
n_sources = 2; %为什么脑源数量设置为2
load('../anatomy/fs_cortex_20k_inflated.mat'); %膨胀后的大脑皮层网格模型
load('../anatomy/fs_cortex_20k.mat');  %原始的高分辨率大脑皮层网格模型，其中 20k 表示大脑表面包含了 20,000 个顶点。
load('../anatomy/fs_cortex_20k_region_mapping.mat'); %大脑皮层的区域映射信息
load('../anatomy/dis_matrix_fs_20k.mat'); %20,000个顶点的皮层网格距离矩阵
% when load mat in python, python cannot read nan properly, so use a magic number to represent nan when saving
NAN_NUMBER = 15213; 
MAX_SIZE = 70;
if train
    nper = 100;                                                            % Number of nmm spike samples  这个是怎么确认的
    n_data = 40;
    n_iter = 48;                                                           % The number of variations in each source center
    ds_type = 'train';
else
    nper = 10;   % 这个尖峰数量为什么会定死 那边每个文件夹中生成的尖峰数量都不相同 我觉得是运行的区域的数量
    n_data = 1;
    n_iter = 3;
    ds_type = 'test';
end
%% ========================================================================
%=============== Generate Source Patch 生成脑皮层上的源区域====================================
%% ======== Region Growing Get Candidate Source Regions ===================
% 下面就是使用区域种植法(Region Growing Method)
selected_region_all = cell(994, 1);                                          
for i=1:994   %这个是遍历每个区域，对每个区域操作
    % get source direction
    % 每个区域选择的脑源区域索引--下面循环的轮数x70 是其矩阵形状
    selected_region_all{i} = [];
    region_id =i;
    all_nb = cell(1,4); %这个参数未知 为什么生成数量为4的细胞数组 这个应该是使用区域种植法迭代的次数
    %nbs参数 是fs_cortex_20k_region_mapping.mat文件中的 代表的是 每个区域的邻居
    all_nb{1} = find_nb_rg(nbs, region_id, region_id);                     % first layer regions
    all_nb{2} = find_nb_rg(nbs, all_nb{1}, [region_id, all_nb{1}]);        % second layer regions
    % centre是fs_cortex_20k_inflated.mat的数组 应该是区域中心点的坐标位置； v0 应该返回的是num_nbx3的数组
    v0 = get_direction(centre(region_id, :),centre(all_nb{1}, :));       % direction between the center region and first layer neighbors
    % 下面这个参数是干什么的 创建了num_nbx1的全0数组
    angs = zeros(size(v0,1),1);
    for k=1:size(v0,1) % 这个for循环计算两个单位向量之间的夹角的角度                                                     
        CosTheta = max(min(dot(v0(1,:),v0(k,:)),1),-1); %dot用于计算两个向量的点积(内积)
        % real用于提取复数的实部。
        angs(k) = real(acosd(CosTheta)); %acosd是计算给定输入值的反余弦值 以度为单位返回结果
    end
    % angs得到每个单位向量和第一个单位向量的夹角的角度
    [~,ind] = sort(angs); % 将得到的角度进行排序 得到其在原数组的索引(index)
    % ceil是用来向上取整的 取前一半的索引
    ind = ind(1:ceil(length(angs)/2));                                     % directions to grow the region
    % second layer neighbours
    for iter = 1:5  % 这的iter又表示的是什么意思 不知道里面套个for循环干嘛 里面又没用到iter这个变量
        all_rg = cell(1,4); %这个就是把all_nb复制了一遍
        for k=1:length(ind) 
            ii = ind(k); 
            all_rg(1:2) = all_nb(1:2); 
            v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :)); % 将计算的索引一个个遍历其单位向量
            all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0));
            [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_rg{2}]);
            final_r = setdiff([region_id, all_nb{1}, all_rg{2} add_rg],rm_rg, 'stable')-1; % 这为什么要把区域id-1? 应该是为了在python中好操作
            selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
        end
    end
    % third layer neighbours
    for iter = 1:5
        all_rg = cell(1,4);
        for k=1:length(ind)
            ii = ind(k); 
            all_rg(1:2) = all_nb(1:2); 
            v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :));
            all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0.1));
            all_rg{3} = find_nb_rg(nbs, all_rg{2}, [region_id, all_rg{1}, all_rg{2}]);  
            all_rg{3} = all_rg{3}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{3}, :),1,-0.15));
            [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_rg{2},all_rg{3}]);
            final_r = setdiff([region_id, all_nb{1}, all_rg{2} all_rg{3} add_rg],rm_rg, 'stable')-1;
            selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
        end
    end
%     % fourth neighbours
%     for iter = 1:5
%     all_rg = cell(1,4);
%     for k=1:length(ind)
%         ii = ind(k); 
%         all_rg(1:2) = all_nb(1:2); 
%         v = get_direction(centre(region_id, :),centre(all_rg{1}(ii), :));
%         all_rg{2} = all_rg{2}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{2}, :),1,0.2));
%         all_rg{3} = find_nb_rg(nbs, all_rg{2}, [region_id, all_rg{1}, all_rg{2}]);  
%         all_rg{3} = all_rg{3}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{3}, :),1,-0.1));
%         all_rg{4} = find_nb_rg(nbs, all_rg{3}, [region_id, all_rg{1}, all_rg{2},  all_rg{3}]); 
%         all_rg{4} = all_rg{4}(get_region_with_dir(v, centre(region_id, :), centre(all_rg{4}, :),1,-0.35));
%         [add_rg, rm_rg] = smooth_region(nbs, [region_id,all_nb{1},all_nb{2},all_rg{3},all_rg{4}]);
%         final_r = setdiff([region_id, all_nb{1}, all_nb{2}, all_rg{3},all_rg{4}, add_rg],rm_rg, 'stable')-1;
%         if length(final_r) < 71
%             selected_region_all{i} = [selected_region_all{i};final_r NAN_NUMBER*ones(1,MAX_SIZE-length(final_r))];
%         end
%     end
%     end
end
%% ======== Get Region Center for Each Sample =============================
% 通过设置 n_iter 和 n_sources 两个变量生成源区域的随机排列，供后续的源区域选取和配置使用。
%n_iter是源中心变异的数量 n_sources是什么变量
selected_region = NAN_NUMBER*ones(994*n_iter, n_sources, MAX_SIZE); 
%n_iter * (n_sources - 1) 是因为代码需要为 每个源区域（除了第一个源）生成 n_iter 个不同的随机排列
n_iter_list = nan(n_iter*(n_sources-1), 994); %这里改一下 原来n_iter_list = nan(n_iter*(n_sources-1), 994);
for i = 1:n_iter
    for k=1:(n_sources-1) %遍历行n_iter_list 生成随机排列
        n_iter_list(i+(k-1)*n_iter,:) = randperm(994);
    end
end
% 这里报错 因为n_iter和n_sources这两个变量 这应该是要有一行为正常序列
n_iter_list(n_iter+1,:) = 1:994;  %这里先把n_iter+1改为n_iter 因为这会超过数组索引
%% ======== Build Source Patch ============================================
for kk = 1:n_iter
    for ii  = 1:994 
        %-------------------处理第一个源--------------------------%
        idx = 994*(kk-1) + ii; % 这在循环里是1到n_iterx994
        tr = selected_region_all{ii};
        if kk <= size(tr, 1) && train
            selected_region(idx,1,:) = tr(kk,:);  % selected_region就是最后的结果
        else
            %随机选择对应区域通过区域种植产生的不同结果 做为最终结果的激活斑块
            selected_region(idx,1,:) = tr(randi([1,size(tr,1)],1,1),:); 
        end
        %-----------------------------------------------------------%
        for k=2:n_sources % 处理剩下的源
            tr = selected_region_all{n_iter_list(kk+n_iter*(k-2),ii)};
            selected_region(idx,k,:) = tr(randi([1,size(tr,1)],1,1),:);
        end
    end
end
%数据重塑和排列
selected_region_raw = selected_region;
selected_region = reshape(permute(selected_region_raw, [3,2,1]), MAX_SIZE*n_sources, 994, n_iter);
selected_region = permute(selected_region,[1,3,2]);
selected_region = reshape(repmat(selected_region, 4, 1, 1), MAX_SIZE, n_sources, []);  % 4 SNR levels
selected_region = permute(selected_region,[3,2,1]);
%% SAVE
dataset_name = 'source1';
save(['../source/' ds_type '_sample_' dataset_name '.mat'], 'selected_region')
%% ========================================================================
%=============== Generate Other Parameters=================================
%% NMM Signal Waveform
random_samples = randi([1,nper],994*n_iter*4,n_sources);                 % the waveform index for each source
nmm_idx = (selected_region(:,:,1)+1)*nper + random_samples + 1; 
save(['../source/' ds_type '_sample_' dataset_name '.mat'],'nmm_idx', 'random_samples',  '-append')
%% SNR
current_snr = reshape(repmat(5:5:20,n_iter*994,1)',[],1); 
save(['../source/' ds_type '_sample_' dataset_name '.mat'],'current_snr', '-append')
%% Scaling Factor
load('../anatomy/leadfield_75_20k.mat');
gt = load(['../source/' ds_type '_sample_' dataset_name '.mat']);
scale_ratio = [];
n_source = size(gt.selected_region, 2);
for i=1:size(gt.selected_region, 1)
    for k=1:n_source
        a = gt.selected_region(i,k,:);
        a = a(:); % 压为一维向量
        a(a>1000) = []; % 去掉值为15213的最高值 也就是没用的数据 保留剩下的所选择的邻居区域id
        if train    
            scale_ratio(i,k,:) = find_alpha(a+1, random_samples(i, k), fwd, 10:2:20); 
            
        else
            %尖峰数据不是都不固定的吗 为什么要把随机的数传入去找对应的尖峰数据 那不很有可能没有？
            scale_ratio(i,k,:) = find_alpha(a+1, random_samples(i, k), fwd, [10,15]);
        end
    end
end
save(['../source/' ds_type '_sample_' dataset_name '.mat'], 'scale_ratio', '-append')
%% Change Source Magnitude 
clear mag_change
point_05 = [40, 60];  % 45,35                                               % Magnitude falls to half of the centre region
point_05 = randi(point_05);
sigma = 0.8493*point_05;
mag_change = [];
for i=1:size(gt.selected_region,1)
    for k=1:n_sources
        rg = gt.selected_region(i,k,:);
        rg(rg>1000) = [];
        dis2centre = raw_dis_matrix(rg(1)+1,rg+1);
        mag_change(i,k,:) = [exp(-dis2centre.^2/(2*sigma^2)) NAN_NUMBER*ones(1,size(gt.selected_region,3)-length(rg))];
    end
end
save(['../source/' ds_type '_sample_' dataset_name '.mat'], 'mag_change', '-append')
%%




function alpha = find_alpha(region_id, nmm_idx, fwd, target_SNR)
% Re-scaling NMM channels in source channels
%
% INPUTS:
%     - region_id  : source regions, start at 1
%     - nmm_idx    : load nmm data
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - target_SNR : set snr between signal and the background activity.
% OUTPUTS:
%     - alpha      : the scaling factor for one patch source

load(['../source/nmm_spikes/a' int2str(region_id(1)-1) '/nmm_' int2str(nmm_idx) '.mat'])
spike_shape = data(:,region_id(1))/max(data(:,region_id(1)));
[~, peak_time] = max(spike_shape);
data(:, region_id) = repmat(spike_shape,1,length(region_id));
[Ps, Pn, ~] = calcualate_SNR(data, fwd, region_id, max(peak_time-50,0):max(peak_time+50,500));
alpha = sqrt(10.^(target_SNR./10).*Pn./Ps);
end


function [Ps, Pn, cur_snr] = calcualate_SNR(nmm, fwd, region_id, spike_ind)
% Caculate SNR at sensor space.
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - spike_ind  : index to calculate the spike snr
% OUTPUTS:
%     - Ps         : signal power
%     - Pn         : noise power
%     - cur_snr    : current SNR in dB

    sig_eeg = (fwd(:, region_id)*nmm(:, region_id)')';   % time * channel
    sig_eeg_rm = sig_eeg - mean(sig_eeg, 1);
    dd = 1:size(nmm,2);
    dd(region_id) = [];
    noise_eeg = (fwd(:,dd)*nmm(:,dd)')';
    noise_eeg_rm = noise_eeg - mean(noise_eeg, 1);

    Ps = norm(sig_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    Pn = norm(noise_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    cur_snr = 10*log10(Ps/Pn);
end


function v = get_direction(a,b)
% Calculate direction between two point
% INPUTS:
%     - a,b        : points in 3D; size 1*3
% OUTPUTS:
%     - v          : direction between two points; size 1*3
v = b-a;
v = v./mynorm(v,2); %在matlab中./是指对数组元素进行逐元素除法
end


function rg = find_nb_rg(nbs, centre_rg, prev_layers)
% Find the neighbouring regions of the centre region  
% 找出中心区域的邻近区域
% INPUTS:
%     - nbs           : neighbour regions for each cortical region; 1*994
%     每个皮质去的相邻区域
%     - centre_rg     : centre regions  中心区域
%     - prev_layers   : regions in inner layers   内层区域
% OUTPUTS:
%     - rg            : neighbouring regions
rg = unique(cell2mat(nbs(centre_rg)));
rg(ismember(rg, prev_layers)) = [];
end


function [selected_rg] = get_region_with_dir(v, region_centre, nb_points, ratio, bias)
% Select region given the region growing direction  根据区域的生长方向选择区域
% INPUTS:
%     - v             : region growing direction
%     - region_centre : centre region in 3D 
%     - nb_points     : neighbour region in 3D 
%     - ratio, bias   : adjust the probability of selecting neighbour
%                       regions (numbers decided by trial and error)
% OUTPUTS:
%     - selected_rg   : selected neighbouring regions

    v2 = get_direction(region_centre, nb_points);                          % direction between center region and neighbour regions
    dir_range = abs(v2*v');                                                % dot product between region growing direction and all neighbouring directions
    dir_range = ratio*((dir_range-min(dir_range))/(max(dir_range) - min(dir_range))) + bias;  % the probability of selecting neighbour regions
%     dir_range = 0.5                                                      % Equal probability for all directions
    selected_rg = rand(length(dir_range),1) < dir_range;
end



function [add_rg, rm_rg] = smooth_region(nbs, current_regions)
% Clean up the current selected regions; since we randomly select the
% neighbouring regions, there could be "holes" in the source patch. We add
% the regions where all its neighbours are in the current source patch; and
% remove the regions where no neighbours is in current source patch;
% INPUTS:
%     - nbs             : neighbour regions for each cortical region; 1*994
%     - current_regions : selected regions 
% OUTPUTS:
%     - selected_rg   : selected neighbouring regions
    add_rg = []; %加入的区域
    rm_rg = [];  %移除的区域
    all_final_nb = find_nb_rg(nbs, current_regions, []); 
    %setdiff 函数用于查找两个数组之间的差集
    all_final_nb = setdiff(all_final_nb, current_regions);
    for i=1:length(all_final_nb)
        current_rg = all_final_nb(i);
        % intersect用于找到两个数组的公共元素
        % 条件判断：如果 current_rg 的大部分邻居（大于总邻居数量 - 2）都在 current_regions 中，则该区域应被视为“空洞”并加入源区域 add_rg
        if length(intersect(current_regions, nbs{current_rg})) > length(nbs{current_rg})-2
            add_rg = [add_rg current_rg];
        end
    end
    for i=1:length(current_regions)
        current_rg = current_regions(i);
        %条件判断：如果 current_rg 在 current_regions 和 add_rg 中的邻居只有一个
        % （即 length(intersect([current_regions, add_rg], nbs{current_rg})) == 1），
        % 则表示 current_rg 是孤立区域，将其加入移除列表 rm_rg。
        if length(intersect([current_regions,add_rg], nbs{current_rg})) == 1
            rm_rg = [rm_rg current_rg];
        end
    end
end


function nn = mynorm(x, varargin)
% Calcualte norm over the rows/columns of a matrix  计算矩阵的行/列的范数
% x: defalut, d*n, calculated the norm over each column
% the second input is the dimension
if nargin > 1
    nn = sqrt(sum(x.^2,varargin{1}));
else
    nn = sqrt(sum(x.^2, 1));
end

end