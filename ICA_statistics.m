%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% ICA statistics - calcuates how many components were excluded
%
% Nele Felicitas Werner (2023)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generic vs. individual model of the encoding model 
clear all
close all
clc

tmp = matlab.desktop.editor.getActive; 
%basepath = fileparts(tmp.Filename);
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata/';
addpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/mTRF_2.3/';

elecs = { 'Fp1','Fp2','F9','F7','F3','Fz','F4','F8','F10','FC5','FC6','T7','C3','Cz','C4','T8',...
'TP9','CP5','CP6','TP10','P7','P3','PZ','P4','P8','O1','O2'};

ROI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];

% participants without noise
participants = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1', '1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};

% initialize a variable to store the counts
n_entries = zeros(length(participants), 1);

% reads entries of ICA components per participant
for p = 1:length(participants)
   load([datapath '/preprocessed/' 'BadComponents_vp' num2str(participants{p}) '.mat']);
   n_entries(p) = length(bad_components);
end

% calculate the range of  excluded ICA components
range = range(n_entries)

% calculate the mean of excluded ICA components per participant
mean_entries = mean(n_entries);
