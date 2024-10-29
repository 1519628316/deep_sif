filename='../source/raw_nmm/a0/mean_iter_0_a_iter_0_1.mat';
%filename2='../source/nmm_spikes/a0/nmm_2.mat';
%load(filename2);
%[time,channels]=size(data);
%x_data=linspace(0,10000,time);
load(filename);
channel=3;
figure;
%plot(all_time',all_data(:,channel)');
plot(time,data(:,channel))
grid on;