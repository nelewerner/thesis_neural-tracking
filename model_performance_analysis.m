%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%
% Script, that computes and plots the model performance of all models
% (encoding individual, encoding generic, decoding individual and 
% decoding generic model) by using the Pearson's r coefficient 
%
% Nele Felicitas Werner (2023)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

tmp = matlab.desktop.editor.getActive; 
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata/';
addpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/mTRF_2.3/';

elecs = { 'Fp1','Fp2','F9','F7','F3','Fz','F4','F8','F10','FC5','FC6','T7','C3','Cz','C4','T8',...
'TP9','CP5','CP6','TP10','P7','P3','PZ','P4','P8','O1','O2'};

ROI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];   
% Auditory ROI: F3, FZ, F4, FC5, FC6

% mother condition - participants without noise 
% (n1/n2 = session 1 or 2)
participants_m = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger condition - participants without noise
participants_s = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};

%% ENCCODING MODEL 
% loading data of the encoding individual model 

% mother condition (m = mother, r = pearson's r coefficient)
for p=1:length(participants_m)
   load ([datapath '/MTRF_m/' 'trf_individual_vp' num2str(participants_m{p}) '.mat']);
   stats_m_r(p,:,:) = stats.r';
  
end

% stranger condition (s = stranger, r = pearson's r coefficient)
for p=1:length(participants_s)
   load ([datapath '/MTRF_s/' 'trf_individual_vp' num2str(participants_s{p}) '.mat']);
   stats_s_r(p,:,:) = stats.r';
  
end

% concatenating both conditions
% since we want to compare the performance of the individual model regardless of the condition
stats_s_r = stats_s_r';
stats_m_r = stats_m_r';
en_in_all_r = [stats_m_r stats_s_r];

% calculating the mean for the whole data of the encoding individual model
mean_en_in_r = squeeze(mean(en_in_all_r));

clearvars -except en_in_all_r en_in_all_err mean_en_in_r mean_en_in_err datapath addpath participants_m participants_s 

%% loading data of the encoding generic model 

% mother condition (m = mother, r = pearson's r coefficient)
% the last participant contains the data of all previous participants
load ([datapath '/MTRF_m/' 'trf_generic_vp46_n1' '.mat']);
STATS_m = STATS.r';

% stranger condition (s = stranger, r = pearson's r coefficient)
load ([datapath '/MTRF_s/' 'trf_generic_vp45_n1' '.mat']);
STATS_s = STATS.r';

% concatenating both condition
en_gen_all_r = [STATS_m STATS_s];

% calculating the mean for the encoding generic model 
mean_en_gen_r =mean(en_gen_all_r);

clearvars -except  en_gen_all_r_plot en_in_all_r en_in_all_err mean_en_in_r mean_en_in_err datapath addpath participants_m participants_s en_gen_all_r mean_en_gen_r

%% DECODING MODEL

% loading data decoding individual model
% mother - participants without noise
participants_m = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger - participants without noise
participants_s = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};

% mother condition (m = mother, r = pearson's r coefficient)
for p=1:length(participants_m)
   load ([datapath '/Decoder_m/' 'decoder_individual_vp' num2str(participants_m{p}) '.mat']);
   dec_m_r(p,:,:) = stats.r';
  
end

% stranger condition (s = stranger, r = pearson's r coefficient)
for p=1:length(participants_s)
   load ([datapath '/Decoder_s/' 'decoder_individual_vp' num2str(participants_s{p}) '.mat']);
   dec_s_r(p,:,:) = stats.r';

end

% concatenating both models 
dec_m_r = dec_m_r';
dec_s_r = dec_s_r';
de_in_all_r = [dec_m_r dec_s_r];

% calculating the mean over all participants for the decoding individual model 
mean_de_in_r = squeeze(mean(de_in_all_r));

clearvars -except en_gen_all_r_plot en_in_all_r en_in_all_err mean_en_in_r mean_en_in_err datapath addpath participants_m participants_s en_gen_all_r mean_en_gen_r de_in_all_r de_in_all_err mean_de_in_r mean_de_in_err en_in_all_r_plot

%% loading the data of the decoding generic model

% mother condition (m = mother, r = pearson's r coefficient)
% the last participant contains the data of all previous participants
load ([datapath '/Decoder_m/' 'Decoder_generic_vp46_n1' '.mat']);
DEC_m = STATS.r;

% stranger condition (s = stranger, r = pearson's r coefficient)
load ([datapath '/Decoder_s/' 'Decoder_generic_vp45_n1' '.mat']);
DEC_s = STATS.r;

% concatenating both models 
de_gen_all_r = [DEC_m DEC_s];

% calculating the mean over all 
mean_de_gen_r =mean(de_gen_all_r);

clearvars -except en_gen_all_r_plot en_in_all_r en_in_all_err mean_en_in_r mean_en_in_err datapath addpath participants_m participants_s en_gen_all_r mean_en_gen_r de_in_all_r de_in_all_err mean_de_in_r mean_de_in_err de_gen_all_r mean_de_gen_r en_in_all_r_plot

%% Calculating Confidence Interval of all models

% mean over electrodes of the encoding model 
mean_en_in_r_2 = mean(mean_en_in_r);
mean_en_gen_r_2 = mean(mean_en_gen_r);

% calculating standard deviation
std_en_in = std(mean_en_in_r);
std_en_gen = std(mean_en_gen_r);
std_de_in = std(de_in_all_r);
std_de_gen = std(de_gen_all_r); 

% set confidence level (e.g., 95%)
conf_level = 0.95;

% calculate degrees of freedom, for all models the same
df = length(en_in_all_r) - 1;

% calculate t-value for given confidence level and degrees of freedom
t_val = tinv(1 - (1 - conf_level)/2, df);

% calculate margin of error
margin_err_1 = t_val * (std_en_in / sqrt(length(std_en_in)));
margin_err_2 = t_val * (std_en_gen / sqrt(length(std_en_gen)));
margin_err_3 = t_val * (std_de_in / sqrt(length(std_de_in)));
margin_err_4 = t_val * (std_de_gen / sqrt(length(std_de_gen)));

% calculate lower and upper bounds of confidence interval
lower_bound_1 = mean_en_in_r_2 - margin_err_1;
upper_bound_1 = mean_en_in_r_2 + margin_err_1;

lower_bound_2 = mean_en_gen_r_2 - margin_err_2;
upper_bound_2 = mean_en_gen_r_2 + margin_err_2;

lower_bound_3 = mean_de_in_r - margin_err_3;
upper_bound_3 = mean_de_in_r + margin_err_3;

lower_bound_4 = mean_de_gen_r - margin_err_4;
upper_bound_4 = mean_de_gen_r + margin_err_4;

% print results
fprintf('ENCODING INDIVIDUAL mean: %.4f\n', mean_en_in_r_2);
fprintf('Confidence interval: [%.4f, %.4f]\n', lower_bound_1, upper_bound_1);

% print results
fprintf('ENCODING GENERIC mean: %.4f\n', mean_en_gen_r_2);
fprintf('Confidence interval: [%.4f, %.4f]\n', lower_bound_2, upper_bound_2);

% print results
fprintf('DECODING INDIVIDUAL mean: %.4f\n', mean_de_in_r);
fprintf('Confidence interval: [%.4f, %.4f]\n', lower_bound_3, upper_bound_3);

% print results
fprintf('DECODING GENERIC mean: %.4f\n', mean_de_gen_r);
fprintf('Confidence interval: [%.4f, %.4f]\n', lower_bound_4, upper_bound_4);

%% PlOTTING Scatter Plot

% Data points
x1 = 0.5;
x2 = 1;
x3 = 1.5;
x4 = 2;

% Plotting
figure();

% ENCODING INDIVIDUAL 
scatter(x1, mean_en_in_r, 'Marker', 'o', 'MarkerEdgeColor', [0.6 0.6 1], 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
% plot black dot at mean of y1
hold on
scatter(x1 + 0.05, mean_en_in_r_2, 'Marker', 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'black', 'LineWidth', 0.5);
errorbar(x1 + 0.05, mean_en_in_r_2, margin_err_1, margin_err_1, 'k', 'LineWidth', 0.5, 'CapSize', 8);

% ENCODING GENERIC
scatter(x2, mean_en_gen_r, 'Marker', 'o', 'MarkerEdgeColor', [0 0 1], 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
hold on
scatter(x2 + 0.05, mean_en_gen_r_2, 'Marker', 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'black', 'LineWidth', 0.5);
errorbar(x2 + 0.05, mean_en_gen_r_2, margin_err_2, margin_err_2, 'k', 'LineWidth', 0.5, 'CapSize', 8);

% DECODING INDIVIDUAL
scatter(x3, de_in_all_r, 'Marker', 'o', 'MarkerEdgeColor', [0.6 0.6 1], 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
errorbar(x3 + 0.05, mean_de_in_r, margin_err_3, margin_err_3, 'k', 'LineWidth', 0.5, 'CapSize', 8);
hold on
scatter(x3 + 0.05, mean_de_in_r, 'Marker', 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'black', 'LineWidth', 0.5);
% DECODING GENERIC
scatter(x4, de_gen_all_r, 'Marker', 'o', 'MarkerEdgeColor', [0 0 1], 'MarkerFaceColor', 'none', 'LineWidth', 0.5);
errorbar(x4 + 0.05, mean_de_gen_r, margin_err_4, margin_err_4, 'k', 'LineWidth', 0.5, 'CapSize', 8);
hold on
scatter(x4 + 0.05, mean_de_gen_r, 'Marker', 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'black', 'LineWidth', 0.5);


% Formatting plot
xlim([0 2.5]);
%ylim([-0.1 1.2]);
xticks([0.5 1.75]);
xticklabels({'Encoding Model', 'Decoding Model'});
ylabel('Accuracy');


% Displaying plot
hold off;

exportgraphics(gcf, [datapath '/Plots/stats-accuracy_scatter_CI95.png'],'BackgroundColor','none','ContentType','vector')

%% PLOTTING BOX PLOT 

% Box Plot of mean performance of each model (mean pearson's r correlation) 
acc_en_ind = mean(mean_en_in_r)
acc_en_gen = mean(mean_en_gen_r)
acc_dec_ind = mean(mean_de_in_r)
acc_dec_gen = mean(mean_de_gen_r)

% Plot prediction accuracy
f5=figure()
bar(1, acc_en_ind, 'facecolor', [0.6 0.6 1])
hold on
bar(2, acc_en_gen, 'facecolor', [0 0 1])
hold on
bar(3, acc_dec_ind, 'facecolor', [0.6 0.6 1])
hold on
bar(4, acc_dec_gen, 'facecolor', [0 0 1])
hold off

% set labels for plot
set(gca,'xtick',1:4,'xticklabel',{'Encoding' 'Encoding' 'Decoding' 'Decoding'})
axis square
title('Model Performance')
legend('Individual', 'Generic');
ylabel('Prediction accuracy [r]')

exportgraphics(gcf, [datapath '/Plots/stats-accuracy.png'],'BackgroundColor','none','ContentType','vector')
