function process_raw_nmm(varargin)
% Scan through the raw NMM data to find the spike data

% %%%%%%%%%%%%%% SETUP PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p = inputParser;
addParameter(p,'filename','spikes',@ischar);   %第三个参数是 spikes
addParameter(p,'leadfield_name','leadfield_75_20k.mat', @ischar); %这个参数在哪用了
parse(p, varargin{:})
filename = p.Results.filename;
headmodel = load(['../anatomy/' p.Results.leadfield_name]);
fwd = headmodel.fwd;
savefile_path = '../source/';

% -------------------------------------------------------------------------
iter_list = 0:2;   % the iter during NMM generation.  把iter_list改为0:4 原来为0:2
previous_iter_spike_num = zeros(1, 994);
for i_iter = 1:length(iter_list)    % 1-3 i_iter的值
    iter = iter_list(i_iter); % iter的值 为 0-2  这代码写的无意义
    %------ 这是因为我更改了部分数值 避免bug 需要添加的代码------%
    % if iter>=3
    %     continue;
    % end
    %--------------------------------------------------------%
    if isempty(dir([savefile_path 'nmm_' filename '/clip_info/iter' int2str(iter)]))
       mkdir([savefile_path 'nmm_' filename '/clip_info/iter' int2str(iter)])
    end

    % ------- Resume running if the process was interupted ----------------
    done = dir([savefile_path 'nmm_' filename '/clip_info/iter' int2str(iter) '/iter_' int2str(iter) '_i_*']);
    finished_regions = zeros(1, length(done));
    for i = 1:length(done)
        finished_regions(i) = str2num(done(i).name(10:end-3));
    end                       %由于python代码的改变 这将1:994 变为 1:5
    remaining_regions = setdiff(1:5, finished_regions+1);  % 得到还未处理的区域id
    if isempty(remaining_regions)
        continue;
    end

    % -------- start the main progress -----------------------------------%
    for ii = 1:length(remaining_regions)     %原来length(remaining_regions)是1                       % Change iteration to the num of NMM regions you want to generate
    %for循环只执行1次
        i = remaining_regions(ii);
        % creat folders to save nmm files
        if isempty(dir([savefile_path 'nmm_' filename '/a' int2str(i-1)]))
            mkdir([savefile_path 'nmm_' filename '/a' int2str(i-1)])
        end

        fn = [savefile_path 'raw_nmm/a' int2str(i-1) '/mean_iter_' int2str(iter) '_a_iter_' int2str(i-1)];
        if isfile([fn '_ds.mat'])                                          % saved downsampled data before
           raw_data = load([fn '_ds.mat']);
           nmm = raw_data.all_data;
        else
           sub_iter_nmm_files = dir([fn '_*.mat']);
           all_data = [];
           all_time = [];
           for sub_iter_i = 1:length(sub_iter_nmm_files)
               % 将目录和文件名字拼接
               d = load([sub_iter_nmm_files(sub_iter_i).folder '\' sub_iter_nmm_files(sub_iter_i).name]);
               all_data = [all_data; d.data]; % 垂直串联操作 将d.data的数据附加到all_data末尾
               all_time = [all_time;d.time']; 
           end
           all_data = all_data(1001:end,:);    %提取2秒的数据                            % Remove the unconverged beginning of the sample 
           all_time = all_time(1001:end);
           all_data = downsample(all_data, 4);  %进行下采样操作 得到每秒500个样本
           all_time = downsample(all_time, 4);
           % 下面的代码 在python中已经处理过了 应该可以不用运行
           %all_data(:,[8,326,922,950]) = all_data(:,[995,998,997,996]);              % remove empty NMM row
           %all_data = all_data(:, 1:994);  % 为什么只去前994个 还是只是把后面几个挪到前面了
           save([fn '_ds.mat'],'all_data','all_time')
           nmm = all_data;
        end
        
        % 下面函数是处理原始tvb输出数据 以找到尖峰峰值的时间
        %应该看看这个函数 传出的spike_time 的值的范围在0-6000  还是在0-500 应该是前者；
        %其返回的值是行向量(spike_time和spike_chan)
        [spike_time, spike_chan] = find_spike_time(nmm);                   % Process raw tvb output to find the spike peak time
        
        % ----------- select the spikes we want to extract ---------------%
        % rule1是bool数组 尖峰通道id和震源id相同的规则
        rule1 =  (spike_chan == i);  % i代表震源区的id                      % there is spike in the source region
        % 计算尖峰时间片段的起始时间
        start_time = floor(spike_time(rule1)/500) * 500 + 1;               % there is no source in other region in the clip
        % 计算900个不同的起始时间
        clear_ind = repmat(start_time, [900, 1]) + (-200:699)';            % 900 * num_spike
        % 下面查看是否有别的脑源id 在这段区间内也产生了尖峰
        rule2 = (sum(ismember(clear_ind, spike_time(~rule1)), 1) == 0);    % there are no other spikes in the clip
        spike_time = spike_time(rule1); %先按照rule1筛选spike_time
        spike_time = spike_time(rule2); %再看产生的标量spike_time是否符合rule2
        %因为2生成的是 1x0 double 但是4生成的是 0x0 double 所以后面报错
        % -----下面是我为了避免报错 添加的代码-----------%
        if isempty(spike_time)
            spike_time=reshape(spike_time,1,0);
        end
        %---------------------------------------------%
         
        % ----------- Optional :  Scale the NMM here----------------------%
        alpha_value = find_alpha(nmm, fwd, i, spike_time, 15); 
        nmm = rescale_nmm_channel(nmm, i, spike_time, alpha_value);       
        % ------------Save Spike NMM Data --------------------------------%
        start_time = floor(spike_time/500) * 500 + 1;
        spike_ind = repmat(start_time, [500, 1]) + (0:499)';       
%         start_time = floor((spike_time+200)/500) * 500 + 1 - 200;        % start time can be changed
%         start_time = max(start_time, 101);
%         spike_ind = repmat(start_time, [500, 1]) + (0:499)';

        nmm_data = reshape(nmm(spike_ind,:), 500, [], size(nmm,2));        % size: time * num_spike * channel
        save_spikes_(nmm_data, [savefile_path 'nmm_' filename '/a' int2str(i-1) '/nmm_'], previous_iter_spike_num(i));
        previous_iter_spike_num(i) = previous_iter_spike_num(i) + length(spike_time);
        % Save something in clip info, so that we can make sure we finish this process
        save_struct = struct();
        save_struct.num_spike = previous_iter_spike_num(i);
        save_struct.spike_time = spike_time;
        parsave([savefile_path 'nmm_' filename '/clip_info/iter' int2str(iter) '/iter_' int2str(iter) '_i_' int2str(i-1) '.mat'], save_struct)
        sprintf(['iter_' int2str(iter) '_i_%d is done\n'], i-1)
    end % END REGION
end % END ITER
end % ENG FUNCTION



%% --------------------- Helper functions ------------------------------ %%
function parsave(fname, mapObj) % 将mapObj(结构体变量，包含了你想要保存的数据)存入fname中
    save(fname,'-struct', 'mapObj');
end


function save_spikes_(spike_data, savefile_path, previous_iter_spike_num)
% Save the spike data into seperate files
% INPUTS: spike_data: time * num_spikes * channel; extracted spike data  被提取的尖峰数据
%         savefile_path: string
    for iii = 1:size(spike_data,2)
        % The raw data
        data = squeeze(spike_data(:,iii,:));
        save([savefile_path int2str(iii+previous_iter_spike_num) '.mat'], 'data', '-v7')
    end
end


function [spike_time, spike_chan] = find_spike_time(nmm)
% Process raw tvb output to find the spike peak time.
%
% INPUTS:
%     - nmm        : (Downsampled) raw tvb output, time * channel
% OUTPUTS:
%     - spike_time : the spike peak time in the (downsampled) NMM data
%     - spike_chan : the spike channel for each spike

    spikes_nmm = nmm;
    spikes_nmm(nmm < 8) = 0;                                               % find the spiking activity stronger than the background
    local_max = islocalmax(spikes_nmm);                                    % find the peak
    [spike_time, spike_chan] = find(local_max);
    [spike_time, sort_ind] = sort(spike_time);
    spike_chan = spike_chan(sort_ind);                                     % sort the activity based on time
    use_ind = (spike_time-249 > 0) & ...                                   % ignore the spikes at the beginning or end of the signal
        (spike_time+250 < size(nmm, 1) & ...
        [1 diff(spike_time)'>100]');                                       % ignore peaks close together for now (will have signals with close peaks in multi-source condition)
    spike_time = spike_time(use_ind)';
    spike_chan = spike_chan(use_ind)';

end



function [alpha] = find_alpha(nmm, fwd, region_id, time_spike, target_SNR)
% Find the scaling factor for the NMM channels.
% 说是通过计算缩放因子alpha来调节神经质量模型数据的信号强度，使其达到指定的目标信噪比(SNR).
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel  time是6000
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - time_spike : spike peak time
%     - target_SNR : set snr between signal and the background activity.
% OUTPUTS:
%     - spike_time : the spike peak time in the (downsampled) NMM data
%     - spike_chan : the spike channel for each spike
%     - alpha      : the scaling factor for one patch source

    % 尖峰索引 找到在尖峰附近符合[0,500]的区间索引
    spike_ind = repmat(time_spike, [200, 1]) + (-99:100)'; 
    %下面那个返回的是一个列向量 其列向量的个数是 200*num_spike
    spike_ind = min(max(spike_ind(:),0), size(nmm,1));                     % make sure the index is not out of range
%     spike_ind = max(0, time_spike-100): max(time_spike+100,size(nmm,1));   % make sure the index is not out of range  
    % 提取峰值形状并重复，没有使用归一化 后面被注释的代码
    spike_shape = nmm(:,region_id(1)); %/max(nmm(:,region_id(1)));
    nmm(:, region_id) = repmat(spike_shape,1,length(region_id)); %这一行又是生成什么数据 运行时 其实没有变化 因为length（region_id）始终为1
    % calculate the scaling factor
    [Ps, Pn, ~] = calcualate_SNR(nmm, fwd, region_id, spike_ind); %计算出信号的功率 和 噪声的功率
    alpha = sqrt(10^(target_SNR/10)*Pn/Ps); %获得信号的放大或缩小的因子
end


function scaled_nmm = rescale_nmm_channel(nmm, region_id, spike_time, alpha_value)
% Re-scaling NMM channels in source channels
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - spike_time : spike peak time
%     - region_id  : source regions, start at 1
%     - alpha_value: scaling factor
% OUTPUTS:
%     - scaled_nmm : scaled NMM in the source region; time * channel

    nmm_rm = nmm - mean(nmm, 1);
    for i=1:length(spike_time)
        sig = nmm_rm(spike_time(i)-249:spike_time(i)+250, region_id);      % one second data around the peak

        thre = 0.1;
        small_ind = find(abs(sig)<thre);                                   % the index that the values are close to 0
        small_ind((small_ind>450) | (small_ind < 50)) = [];
        start_ind = find((small_ind-250)<0);                               % spike start time
        % test 1
        while isempty(start_ind)
            thre = thre+0.05;
            small_ind = find(abs(sig)<thre);
            small_ind((small_ind>450) | (small_ind < 50)) = [];
            start_ind = find((small_ind-250)<0);
        end
        start_sig = small_ind(start_ind(end));

        % test 2
        [~, min_ind] = min(sig(301:400));
        min_ind = min_ind + 301;
        end_ind = find((small_ind-min_ind)>0);
        while isempty(end_ind)
            thre = thre+0.05;
            small_ind = find(abs(sig)<thre);
            small_ind((small_ind>450) | (small_ind < 50)) = [];
            end_ind = find((small_ind-min_ind)>0);
        end
        end_sig = small_ind(end_ind(1));                                   % spike end time

        sig(start_sig:end_sig) = sig(start_sig:end_sig) * alpha_value;     % scale the signal
        nmm_rm(spike_time(i)-249:spike_time(i)+250, region_id) = sig;
    end
    scaled_nmm = nmm_rm + mean(nmm, 1);


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

    sig_eeg = (fwd(:, region_id)*nmm(:, region_id)')';   % time * channel  计算信号eeg
    sig_eeg_rm = sig_eeg - mean(sig_eeg, 1); 
    dd = 1:size(nmm,2); %确定非目标源区域的索引
    dd(region_id) = []; 
    noise_eeg = (fwd(:,dd)*nmm(:,dd)')';
    noise_eeg_rm = noise_eeg - mean(noise_eeg, 1);

    Ps = norm(sig_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind); %norm 是湖是计算给的数据中 Frobenius范数
    Pn = norm(noise_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    cur_snr = 10*log10(Ps/Pn);
end
