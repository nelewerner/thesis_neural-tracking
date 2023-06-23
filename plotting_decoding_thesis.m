%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The following code plots plots the reconstruction of the stimulus     %
% using the data of the decoding model.                                 %
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

% (1) Set parameters
% ----------------------

fs = 500;     % EEG and stimulus sampling rate
tmin = 0;     % start of time window used for training
tmax = 800;   % end of time window used for training
              % NOTE: when optimizing the model for reconstructive performance 
              % time lags should be restricted to periods capturing the 
              % strongest signal in the temporal response function
             
stim_param = 1;   % which stimulus parameter, here we only model the 
                  % auditory onset envelope as stimulus feature
                  
lambda_exp = -7:7; % range of hyperparameters to test
nlambdas = length(lambda_exp);
                           
direction = -1;     % direction of causality (forward=1, backward=-1)


%%
%%%%%%%%%%%%%%
% Read data  %
%%%%%%%%%%%%%%

% using data of one representative participant 
participants_m = {'2_n2'}

% load stimulus data of the participant 
% first part of the decoding model code (train_test_decoding_model.m) to get 
% the exact same stimulus snippet which is also reconstructed 
for p=1:length(participants_m)
    % load input data containing EEG responses and auditory stimulus features    
    load([datapath 'aligned/Inputdata' participants_m{p} '.mat']) 
    
    % normalize EEG responses (Important: time has to be the first dimension to preserve
    % relative differences between the individual EEG channels)
    resp = zscore(eegdata);  
    
    % stimulus feature for modelling (here, we only have one auditory feature)
    stim = stimulusdata(:,stim_param);
    
    % Downsample data to speed up computations
    fs_ds = 64;
    stim = resample(stim,fs_ds,fs);
    resp = resample(resp,fs_ds,fs);
    
    % split data into k folds for nested crossvalidation
    k = 5; % number of folds
    fold_size = ceil(size(stim,1)/k);
    
    % Set aside first fold as testing set for final model evaluation that will
    % not be part of hyperparameter optimization
    idx = 1:min(fold_size*1,size(stim,1));
    stimtest = stim(idx,:);
    resptest = resp(idx,:);
    
end

% loads reconstructed stimulus of the representative participant
for p=1:length(participants_m)
    reconstructed_stimulus = load ([datapath '/Decoder_m/' 'decoder_individual_vp' num2str(participants_m{p}) '.mat']);

end


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot results (for one representative subject) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f2=figure()
% actual stimulus 
plot((1:length(stimtest))/fs_ds,stimtest, linewidth = 1.5);
hold on
% reconstructed stimulus
plot((1:length(reconstructed_stimulus.recon))/fs_ds,reconstructed_stimulus.recon, linewidth = 1.5)
legend('Stimulus [s]', 'Reconstructed stimulus [s]')
xlim([0 7])
ylim([-0.5 2])

exportgraphics(gcf, [datapath '/Plots/stimulus-recon.png'],'BackgroundColor','none','ContentType','vector')

