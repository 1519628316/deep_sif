# python代码-generate_tvb_data.py部分

**对于后面的代码解析 应该想清楚每个文件 通过什么 然后得到了什么文件 文件是描述什么的**

​	The Virtual Brain Simulation

## 主要部分

​	通过传参，生成在源区域下用xx**mean_and_std（均值和标准差）**生成xx区域下第xx时间片段下的matlab数据文件——mean_iter_**0** _a_iter\_**0**\_**0**.mat

​	其默认给了三组不同的均值和标准差:

​	`mean_and_std = np.array([[0.087, 0.08, 0.083], [1, 1.7, 1.5]])`

​	通过TheVirualBrain包中的simulator模块，使用model.JansenRit()函数生成一个 **Jansen-Rit**对象，后面同一个包下的simulator.Simulator函数进行**模拟数据**

​	model Package 是**神经元动力学模型**的集合。

---

```python
jrm = models.JansenRit(A=use_A, mu=np.array(mean_and_std[0][iter_m]),
                       v0=np.array([6.]), p_max=np.array([0.15]), p_min=np.array([0.03]))
						#上面的A、mu、v0、p_max和p_min都是jansen-rit模型的相关参数
```

---

​	中间产生了一些需要使用的变量，integrators是属于simulator Package的，在integrators中定义了一组确定性和随机微分方程的积分方法。

其中integrators.HeunStochastic()：   这是预测校正方法的一个简单例子。它也被称为修正梯形法，它使用欧拉方法作为其预测器。

在integrators包中：dt是积分步长；noise是随机积分器的噪声源。

noise.Additive的作用是：假设源噪声为单位方差的高斯噪声，则会产生标准差为nsig的加性噪声。

```python
phi_n_scaling = (jrm.a * 3.25 * (jrm.p_max - jrm.p_min) * 0.5 * mean_and_std[1][iter_m]) ** 2 / 2.
sigma = np.zeros(6)
sigma[4] = phi_n_scaling

# set the random seed for the random intergrator  --使用下面代码 确保在执行包含随机性的积分计算时，结果是可复现的
randomStream = np.random.mtrand.RandomState(0) #生成创建伪随机数生成器 0是种子
noise_class = noise.Additive(random_stream=randomStream, nsig=sigma) 
integ = integrators.HeunStochastic(dt=2 ** -1, noise=noise_class) #通过上面的参数进行配置
```

---

```pthon
sim = simulator.Simulator(
    model=jrm,
    connectivity=conn,
    coupling=coupling.SigmoidalJansenRit(a=np.array([1.0])),
    integrator=integrators.HeunStochastic(dt=2 ** -1, noise=noise.Additive(nsig=sigma)),
    monitors=(monitors.Raw(),)
).configure()
```

# matlab代码-process_raw_nmm.m部分

Process Raw TVB Data Prepare Training/Testing Dataset

由mean_iter_0_a_iter_0_0.mat等文件产生mean_iter_0_a_iter_0_ds.mat、iter_0_i_0.mat、nmm_1.mat文件

该函数为每个区域提供了提取的1秒nmm片段，大小为500乘以994 (num_of_time_samples乘以num_of_NMM)，保存在源代码中。

应该流程是**将数据进行降采样--寻找所需要的spike_time(满足其rule1和rule2)--对NMM数据进行处理(使其达到目标SNR)--保存NMM数据**

## 主要部分

首先需要搞清楚 产生的三个文件分别代表什么含义？

`mean_iter_0_a_iter_0_ds.mat`

这个是之前保存的下(降)采样后的数据

`iter_0_i_0.mat`

在剪切时保存的一些信息，尖峰数量 以及符合rule1和rule2的时间位置

`nmm_1.mat`

这是所谓nmm文件，即时间x通道，后面的数字是代表检测出来符合规则的尖峰的个数id  提取出来的尖峰

---

**问题1**

由于在./source/nmm_spikes/iter0中的数据对应的是 ./source/nmm_spikes/a0、./source/nmm_spikes/a1、./source/nmm_spikes/a2、./source/nmm_spikes/a3、./source/nmm_spikes/a4

而其他的数据都没有生成所对应的数据，那为啥会生成 例如：./source/nmm_spikes/iter1等文件夹

​	在后面都说清楚了 a后面的数字是指区域id  iter后面数字是指 mean_and_std的组数

---

**问题2**

在函数**find_alpha**中需要搞清楚的局部变量**spike_ind**，这个变量的含义是什么? 在后面的calculate_SNR函数中也在使用。

```matlab
function [alpha] = find_alpha(nmm, fwd, region_id, time_spike, target_SNR)
% Find the scaling factor for the NMM channels.
% 说是通过计算缩放因子alpha来调节神经质量模型数据的信号强度，使其达到指定的目标信噪比(SNR).
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - time_spike : spike peak time
%     - target_SNR : set snr between signal and the background activity.
% OUTPUTS:
%     - spike_time : the spike peak time in the (downsampled) NMM data
%     - spike_chan : the spike channel for each spike
%     - alpha      : the scaling factor for one patch source

    % 尖峰索引 找到在尖峰附近符合[0,500]的区间索引
    spike_ind = repmat(time_spike, [200, 1]) + (-99:100)';  %应该是选取了200大小的窗口
    spike_ind = min(max(spike_ind(:),0), size(nmm,1));                     % make sure the index is not out of range
%     spike_ind = max(0, time_spike-100): max(time_spike+100,size(nmm,1));   % make sure the index is not out of range  
    % 提取峰值形状并重复，没有使用归一化 后面被注释的代码
    spike_shape = nmm(:,region_id(1)); %/max(nmm(:,region_id(1)));
    nmm(:, region_id) = repmat(spike_shape,1,length(region_id));
    % calculate the scaling factor
    [Ps, Pn, ~] = calcualate_SNR(nmm, fwd, region_id, spike_ind); %计算出信号的功率 和 噪声的功率
    alpha = sqrt(10^(target_SNR/10)*Pn/Ps); %获得信号的放大或缩小的因子
end
```
根据函数**calcualate_SNR**的参数说明 是说 **\- spike_ind  : index to calculate the spike snr**,下面那行代码产生每个spike_time时刻上下100左右的时间索引

`spike_ind = min(max(spike_ind(:),0), size(nmm,1)); ` spike_ind应该是200*num_spike的矩阵

还有下面这行代码所做的事情是什么? 其实没啥变化

`nmm(:, region_id) = repmat(spike_shape,1,length(region_id));`

将这些问题解决说不定就能解决问题1

---

**问题3**

对于文件中的其中for循环来说

```
iter_list = 0:2
for i_iter = 1:length(iter_list)    % 1-3 i_iter的值
    iter = iter_list(i_iter); % iter的值 为 0-2  这代码写的无意义
```

这个iter代表的是在python代码中使用的不同组 **mean_and_std的数量**



对于文件中的另一个for循环来说

```matlab
for ii = 1:length(remaining_regions)         % Change iteration to the num of NMM regions you want to generate
	i = remaining_regions(ii);
```

这个**i**代表的是生成用不同脑源的nmm数据中 **脑源的数量**



# matlab代码-generate_sythetic_source.m部分

This function creates 'test_sample_source1.mat' by default, which describes now to load the nmm spikes, how to scale the background noise, etc. This mat file can be used as input training or testing data for `loaders.SpikeEEGBuild` or `loaders.SpikeEEGBuildEval`.

会创建test_sample_source1.mat文件作为训练数据输入

步骤为**Generate Source Patch--**

## 主要部分

---

**问题1** selected_region_all这个变量究竟起什么作用？ 主要是因为他的矩阵有点奇怪

**selected_region_all**是994x1的细胞数组。

而**selected_region_all{i}**是循环的轮数x70的矩阵。

```matlab
selected_region_all = cell(994, 1);  
for i=1:994   %这个是遍历每个区域，对每个区域操作
    % get source direction
    % 每个区域选择的脑源区域索引--下面循环的轮数x70 是其矩阵形状
    selected_region_all{i} = [];
```

如果只产生一行 我可以认为这是标签 就是需要预测的结果--即实际脑源。

```matlab
%% ======== Build Source Patch ============================================
for kk = 1:n_iter
    for ii  = 1:994
        idx = 994*(kk-1) + ii;
        tr = selected_region_all{ii};
        if kk <= size(tr, 1) && train
            selected_region(idx,1,:) = tr(kk,:);
        else
            selected_region(idx,1,:) = tr(randi([1,size(tr,1)],1,1),:);
        end
        for k=2:n_sources
            tr = selected_region_all{n_iter_list(kk+n_iter*(k-2),ii)};
            selected_region(idx,k,:) = tr(randi([1,size(tr,1)],1,1),:);
        end
    end
end
```

**selected_region_all**在这个里面也有用到 看这段代码怎么处理这个数据的。

我觉得是每个区域通过区域种植法 选取的一块激活斑块(patches of activation in the source space)

但是 为什么selected_region_all{i}的列维度为70 行维度为循环的轮数？为什么这么设计？

列维度为70 我认为是限制所选择的激活斑块中的区域数量不得多于70。 只要选取最后一行就能得到最后的结果--即想要的激活斑块。

---

**问题2** n_iter变量的具体含义

虽然他给了说明---即**The number of variations in each source center**

```matlab
NAN_NUMBER = 15213; 
MAX_SIZE = 70;
%n_iter是源中心变异的数量 n_sources是什么变量
selected_region = NAN_NUMBER*ones(994*n_iter, n_sources, MAX_SIZE); 
%n_iter * (n_sources - 1) 是因为代码需要为 每个源区域（除了第一个源）生成 n_iter 个不同的随机排列
n_iter_list = nan(n_iter*(n_sources-1), 994);
```

这个变量绑定了两个变量 搞懂这个 就很容易搞懂别的



---

**问题3** n_sources是什么变量

n_sources 是指源的数量吗 我觉得是就是数据集中 每次选择两个源作为激活源



