%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code plots the multivariate temporal response           %
% function(mTRF) of the encoding model for both conditions of the       %
% Region of Interest (fronto-central area) well as for all electrodes.  %
% A t-test against zero as well as a paired t-test for both conditions  % 
% was conducted.                                                        %
%                                                                       %
% In addition, the mean mTRF for both conditions were illustrated in a  %
% topoplot.                                                             %
%                                                                       %
% Nele Felicitas Werner (2023)                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

clear all
close all
clc

tmp = matlab.desktop.editor.getActive; 
%basepath = fileparts(tmp.Filename);
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata/';
addpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/mTRF_2.3/';

% all electrodes
elecs = { 'Fp1','Fp2','F9','F7','F3','Fz','F4','F8','F10','FC5','FC6','T7','C3','Cz','C4','T8',...
'TP9','CP5','CP6','TP10','P7','P3','PZ','P4','P8','O1','O2'};

% All electrodes 
ROI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];   

% Auditory ROI: F3, FZ, F4, FC5, FC6 
% needs to be commented out when calculating the mTRF of the Region of
% Interest
% ROI = [5,6,7,10,11];

% mother condition (remaining data sets without noise)
participants_m = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger condition (remaining data sets without noise)
participants_s = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};


%%
%%%%%%%%%%%%%%
% Read data  %
%%%%%%%%%%%%%%

%mother condition 
% load data of all participants to one structure
for p=1:length(participants_m)
   load ([datapath '/MTRF_m/' 'trf_individual_vp' num2str(participants_m{p}) '.mat']);
   model_w(p,:,:) = model.w;
   model_b(p,:,:) = model.b;
   model_t(p,:,:) = model.t;
end

% Calculate mean
mean_w = mean(model_w);
mean_b = mean(model_b);
mean_t = mean(model_t);

% create structure for plotting
mean_mother = struct();
mean_mother.w = mean_w;
mean_mother.b = squeeze(mean_b);
mean_mother.t = squeeze(mean_t);
mean_mother.fs = 500;
mean_mother.Dir = 1;
mean_mother.type = 'multi';

% stranger condition
% load data
for p=1:length(participants_s)
   load ([datapath '/MTRF_s/' 'trf_individual_vp' num2str(participants_s{p}) '.mat']);
   model_s_w(p,:,:) = model.w;
   model_s_b(p,:,:) = model.b;
   model_s_t(p,:,:) = model.t;
end

% Calculate mean
mean_s_w = mean(model_s_w);
mean_s_b = mean(model_s_b);
mean_s_t = mean(model_s_t);

% create structure for plotting
mean_stranger = struct();
mean_stranger.w = mean_s_w;
mean_stranger.b = squeeze(mean_s_b);
mean_stranger.t = squeeze(mean_s_t);
mean_stranger.fs = 500;
mean_stranger.Dir = 1;
mean_stranger.type = 'multi';


%%
%%%%%%%%%%%%%%%%%%%%
% t-test analysis  %
%%%%%%%%%%%%%%%%%%%%

% adding 0 such that both conditions have the same length
model_s_w(15,501,28) = 0;

trf_mother = squeeze(mean_w(:,:,ROI));
trf_stranger = squeeze(mean_s_w(:,:,ROI));

trf_mother = trf_mother';
trf_stranger = trf_stranger';

% calculate mean 
mean_trf_stranger = mean(trf_stranger);
mean_trf_mother = mean(trf_mother);

% values between 200-350 ms (P2 response)
trf_mother_P2 = mean_trf_mother(200:350);
trf_stranger_P2 = mean_trf_stranger(200:350);

% paired t-test to analyse differences between conditions
[h, p, ci,stats] = ttest(trf_mother_P2, trf_stranger_P2)
t_values = stats.tstat
%%
% t-test against zero for both conditions for P2
[h, p, ci, stats] = ttest(trf_mother_P2)
[h, p, ci, stats] = ttest(trf_stranger_P2)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% t-test analysis            %
% for encoding generic model %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loading data of the mother condition 
load ([datapath '/MTRF_m/' 'trf_generic_vp46_n1' '.mat']);
generic_model_w = generic_model.w;

% loading data of the stranger condition 
load ([datapath '/MTRF_s/' 'trf_generic_vp45_n1' '.mat']);
generic_model_s_w = generic_model.w;

trf_generic_mother = squeeze(generic_model_w(:,:,ROI));
trf_generic_stranger = squeeze(generic_model_s_w(:,:,ROI));


trf_generic_mother = trf_generic_mother';
trf_generic_stranger = trf_generic_stranger';

% calculate mean for both conditions
mean_trf_generic_stranger = mean(trf_generic_stranger);
mean_trf_generic_mother = mean(trf_generic_mother);

% t-test for generic mother condition P2
trf_generic_mother_P2 = trf_generic_mother(200:350);
[h, p, ci, stats] = ttest(trf_generic_mother_P2)


% t-test for generic stranger condition P2
trf_generic_stranger_P2 = trf_generic_stranger(200:350);
[h, p, ci, stats] = ttest(trf_generic_stranger_P2)


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting mean mTRF for all electrodes for  %
% both conditions                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Region of Interest
ROI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];   

figure()
sampling_rate = 500; % Set the new sampling rate to 500 Hz

t = -200:1/sampling_rate:800; % Create a time vector
x = linspace(min(t), max(t), numel(mean_trf_mother)); % Create an x-axis vector

plot(x, mean_trf_mother, 'linewidth', 1.5, 'color', [0 0.4470 0.7410])
hold on
plot(x, mean_trf_stranger, 'linewidth', 1.5, 'color', [0.8500 0.3250 0.0980])
xlim([-200 800])
title('Temporal Response function - individual model')
subtitle('mother and stranger condition')
legend('mother', 'stranger')
xlabel('[ms]')

% export graphics
exportgraphics(gcf, [datapath '/Plots/mTRF_all-elec.png'],'BackgroundColor','none','ContentType','vector')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting mTRF for fronto-central area (ROI)      %
% with paired t-test and significant differences   % 
% between conditions                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure()
sampling_rate = 500; % Set the new sampling rate to 500 Hz
subplot(2,1,1)
t = -200:1/sampling_rate:800; % Create a time vector
x = linspace(min(t), max(t), numel(mean_trf_mother)); % Create an x-axis vector
subplot(2,1,1)
plot(x, mean_trf_mother, 'linewidth', 1.5, 'color', [0 0.4470 0.7410])
hold on
subplot(2,1,1)
plot(x, mean_trf_stranger, 'linewidth', 1.5, 'color', [0.8500 0.3250 0.0980])
xlim([-200 800])
title('Temporal Response function - individual model')
subtitle('mother and stranger condition')
legend('mother', 'stranger')
xlabel('[ms]')
subplot(2,1,2)
sampling_rate = 500;
t = -200:1/sampling_rate:800; % Create a time vector
x = linspace(min(t), max(t), numel(mean_trf_mother)); 
plot(x, t_values', linewidth=1.5, color = [0.9290 0.6940 0.1250])
hold on
xlim([-200 800])
hold on 
subplot(2,1,2)
sig_idx = find(p < 0.05); % find indices where p < 0.05
scatter(x(sig_idx), -11*ones(size(sig_idx)), '*');
xlim([-200 800])
ylim([-15 8]) % set y-axis limits
legend('t values', 'p < 0.05')
xlabel('[ms]')
ylabel('t values')

exportgraphics(gcf, [datapath '/Plots/mTRF_ROI.png'],'BackgroundColor','none','ContentType','vector')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Topoplots mother and stranger condition  %
% using the mTRF data of the encoding      %
% individual model                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% time vector ranging from -200 ms to 1000 ms in steps of 2 ms.
T=[-200:2:800];

% just any participant for the FT structure
eegdata = [datapath '/preprocessed/PostICA5_n1.mat']; 
load(eegdata);

% mother condition
plotty.label = preproc.label; % cell array of channel labels
plotty.avg = squeeze(mean_mother.w)'; % channel data first channels, second dimension time
plotty.time = T; % vector of time points corresponding to the first dimension of avg 
plotty.dimord = 'chan_time'; % string indicating the order of dimensions in the avg array


% cfg structure
cfg = [];
 cfg.zlim = [-0.00005 0.00005]; % sets the color range for plotting the data
 cfg.parameter = 'avg';
 cfg.fontsize = 12;
 cfg.layout ='infant.lay';
 cfg.marker = 'on';
 cfg.showlabels = 'yes';
 cfg.markersize = 5;
 cfg.markersymbol = 'o';
 cfg.comment = 'xlim';

 f2 = figure('Position', get(0, 'Screensize')); 
 cfg.xlim = [-200:200:800];
 ft_topoplotER(cfg,plotty);
 cmap = cmocean('balance'); 
 colormap(cmap);  
 colorbar;

exportgraphics(gcf, [datapath '/Plots/topoplot-mother.png'],'BackgroundColor','none','ContentType','vector')

% stranger condition
plotty.label = preproc.label;
plotty.avg = squeeze(mean_stranger.w)';
plotty.time = T;
plotty.dimord = 'chan_time';

% cfg structure 
cfg = [];
cfg.zlim = [-0.00005 0.00005];
cfg.parameter = 'avg';
cfg.fontsize = 12;
cfg.layout ='infant.lay';
cfg.marker = 'on';
cfg.showlabels = 'yes';
cfg.markersize = 5;
cfg.markersymbol = 'o';
cfg.comment = 'xlim';

f2 = figure('Position', get(0, 'Screensize'));
 
cfg.xlim = [-200:200:800];
ft_topoplotER(cfg,plotty);

cmap = cmocean('balance'); 
colormap(cmap)
colorbar;

exportgraphics(gcf, [datapath '/Plots/topoplot-stranger.png'],'BackgroundColor','none','ContentType','vector')
