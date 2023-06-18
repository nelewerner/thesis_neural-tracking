% Script to extract the desired feature of the given wave-file
% using feature extraction scripts from Lorenz (?)
%
% Sarah J, May 2021

clear all;

addpath '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/ac_feature_extraction'; % Add folder where ac_feature_extraction is located
audiopath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/soundfiles/'; % Path where the audio-file is located
outpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/soundfiles/converted/';

%participants = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15','16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40','41','42','43','44','45','46'};

for i = 1:length(participants)

[Y Fs_in] = audioread([audiopath 'vp' participants{i} '.wav']);

Fs_out = 500;   % Fs_out-Sampling-frequency of your EEG Signal
Lp_eeg = 10;    % low pass
FC = 2;         % Number of frequency channels
D = 2;          % Domain, 2 -> Cochlea
B = 1;          % Spectral Resolution, 1 -> Broadband
F = 2;          % Feature, 2 -> Onset
Oct = -1;       % Octave-shift of frequency axis, 100-4000 Hz
audio = ac_feature_extraction(Y,Fs_in,Fs_out,Lp_eeg,FC,D,B,F,Oct);

save ([outpath 'extracted_feature' participants{i} '.mat'], 'audio');

end