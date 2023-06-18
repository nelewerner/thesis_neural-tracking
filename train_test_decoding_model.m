%%TRAIN_TEST_DECODING_MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training, tesing and visualization of decoding model using the
%   mTRF toolbox v2.3
%
% Requires:
% - mTRFtoolbox v2.3 (supplied with this tutorial);
%  (https://github.com/mickcrosse/mTRF-Toolbox)
% - Aligned EEG and stimulus feature data found in 'data' folder
% 'Inputdata*.mat'
%
%  Sarah Tune (2021) U Luebeck
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

tmp = matlab.desktop.editor.getActive; 
%basepath = fileparts(tmp.Filename);
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata/';
addpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/mTRF_2.3/';


% tmp = matlab.desktop.editor.getActive;
% basepath = fileparts(tmp.Filename);
% datapath = [basepath '/data/'];
% addpath([basepath '/mTRF_2.3'])

%% (1) Set parameters
% ----------------------

fs = 500; % EEG and stimulus sampling rate

tmin = 0;  % start of time window used for training
tmax = 800;   % end of time window used for training
              % NOTE: when optimizing the model for reconstructive performance 
              % time lags should be restricted to periods capturing the 
              % strongest signal in the temporal response function
             
stim_param = 1;   % which stimulus parameter, here we only model the 
                  % auditory onset envelope as stimulus feature
                  
lambda_exp = -7:7; % range of hyperparameters to test
nlambdas = length(lambda_exp);
                           
direction = -1;     % direction of causality (forward=1, backward=-1)


%% (2) Training and testing using data splits from the same single-subject dataset
% --------------------------------------------------------------------------------

% mother - participants without noise
% participants = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger - participants without noise
% participants = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};


for p=1:length(participants)
    % load input data containing EEG responses and auditory stimulus features    
    load([datapath 'aligned/Inputdata' participants{p} '.mat']) 
    
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
    
    % all other folds are used in the inner cross-validation loop to 
    % optimize the hyperparameter
    stimtrain = cell(k-1,1);
    resptrain = cell(k-1,1);
    
    for i = 1:k-1
        idx = fold_size*(i-1)+1:min(fold_size*i,size(stim,1));
        stimtrain{i,1} = stim(idx,:);
        resptrain{i,1} = resp(idx,:);
    end
    
    % Leave-one-out cross-validation to determine optimal lambda parameter 
    for fold = 1:size(stimtrain,1)
    
        clear stest rtest strain rtrain
        strain = stimtrain;
        rtrain = resptrain;
    
        % remove current fold from training set to keep for validation
        sval = strain{fold};
        rval = rtrain{fold};        
        strain(fold) = [];
        rtrain(fold) = [];
    
        for l = 1:length(lambda_exp)
    
            % set hyperparameter lambda
            lambda = 10.^lambda_exp(l); 
    
            % train model on the remaining training data segments        
            MODEL(fold, l) = mTRFtrain(strain,rtrain,fs_ds,direction,tmin,tmax,lambda);
    
            % test model using left out validation data segment
            RECON{fold,l} = mTRFpredict(sval,rval,MODEL(fold,l));
    
            % compare reconstruction to original stimulus
            [CV.r(fold,l),CV.err(fold,l)] = mTRFevaluate(sval, RECON{fold,l});
        end 
    
    end
    
    %% (3) Determine optimal lambda parameter and test model on left-out data
    % ------------------------------------------------------------------------
    
    % assess reconstruction accuracy (Pearson's r) 
    individ_r = squeeze(mean(CV.r));
    individ_err = squeeze(mean(CV.err));
    
    % Visual inspection of lambda trace
    f1=figure()
    subplot(1,2,1)
    errorbar(1:nlambdas,individ_r,std(CV.r)/sqrt(k-1),'linewidth',2)
    set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
    title('Cross-validated reconstruction accuracy')
    xlabel('Regularization (1\times10^\lambda)')
    ylabel('Pearsons r')
    axis square
    
    subplot(1,2,2)
    errorbar(1:nlambdas,individ_err,std(CV.err)/sqrt(k-1),'linewidth',2)
    set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
    title('Cross-validated mean squared error')
    xlabel('Regularization (1\times10^\lambda)')
    ylabel('Mean squared error (MSE)')
    axis square
    
    %% 
    % here, we select the lambda parameter that maximizes the correlation
    % between original and reconstructed stimulus 
    [max_r, idx_max] = max(mean(CV.r));
    lambda_opt = 10.^mean(lambda_exp(idx_max));
    
    % fit model with optimal lambda value on all of the training folds
    model = mTRFtrain(stimtrain,resptrain,fs_ds,direction,tmin,tmax,lambda_opt);
    
    % test model on left out test data set
    [recon,stats] = mTRFpredict(stimtest,resptest,model);
    
    % also forward transform backward model 
    fwd_model = mTRFtransform(model, resptrain);

    save([datapath '/Decoder_s/decoder_individual_vp' participants{p} '.mat'],'recon','model','stats', 'lambda_opt', 'fwd_model');

    saveas(gcf, [datapath '/Decoder_s/lambda_individual_vp' participants{p} '.png']);
% 
    %% (4) Plot Forward-Transformed Temporal Response Function (TRF)
    % --------------------------------------------------------------------
    
    % exemplary fronto-central channel
    chan=11; % FC6 
    
    % Plot results (for one representative subject)
    f2=figure()
    subplot(4,1,1)
    plot((1:length(stimtest))/fs_ds,stimtest);
    title('Stimulus [s]')
    
    subplot(4,1,2)
    plot((1:length(resptest))/fs_ds,resptest(:,chan))
    title('EEG [s]')
    
    subplot(4,1,3)
    plot(fwd_model.t,fwd_model.w(1,:,chan))
    xlim([0 800])
    title('Foward-transformed Temporal Response Function at Channel FC6 [ms]')
    
    subplot(4,1,4)
    plot((1:length(recon))/fs_ds,recon)
    title('Reconstructed stimulus [s]')

    saveas(gcf, [datapath '/Decoder_s/decoder_individual_vp' participants{p} '.png']);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (5) Training and testing using a generic (subject-independent model)
% --------------------------------------------------------------------

% Note: you can alternatively to skip to the next section and load the 
% results of the cross-validation loop for final model fitting and testing 


% mother - participants without noise
%participants = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger - participants without noise
% participants = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};


% Train model for each participant using all data
for p = 1:length(participants)
    
    % load input data containing EEG responses and auditory stimulus features    
    load([datapath 'aligned/Inputdata' participants{p} '.mat'])

    % normalize EEG responses (Important: time has to be the first dimension to preserve
    % relative differences between the individual EEG channels)
    RESP{p} = zscore(eegdata);  

    % stimulus feature for modelling (here, we only have one auditory feature)
    STIM{p} = stimulusdata(:,stim_param);
    
    % Downsample data to speed up computations
    fs_ds = 64;
    STIM{p} = resample(STIM{p},fs_ds,fs);
    RESP{p} = resample(RESP{p},fs_ds,fs);
    
    % For each lambda value fit model to all data per participant 
    for l = 1:length(lambda_exp)
        
        clear model
        % set hyperparameter lambda
        lambda = 10.^lambda_exp(l); 

        % train model on all data
        model = mTRFtrain(STIM{p},RESP{p},fs_ds,direction,tmin,tmax,lambda);        
              
        % store models from all participants in one structure
        MODELS(p,l,:,:) = model.w;
        CONST(p,l) = model.b;

    end
    
    % keep model structure for creating generic models below
    if p==1
         generic_model = model;
    end
end

% Average models per lambda across n-1 participants, test on left out participant
for p =1:length(participants)
    
    clear models consts
    % remove current participant       
     models = MODELS;
     models(p,:,:,:) = [];
     
     consts = CONST;
     consts(p,:) = [];
    
    % for each lambda
    for l = 1:length(lambda_exp)
        
        % mean across models        
        generic_model.w = squeeze(mean(models(:,l,:,:)));
        generic_model.b = squeeze(mean(consts(:,l,:)));
        
        % test model using left out data segment
        RECON{p,l} = mTRFpredict(STIM{p},RESP{p},generic_model);

        % compare prediction to actual EEG responses
        [CV_generic.r(p,l),CV_generic.err(p, l)] = mTRFevaluate(STIM{p}, RECON{p,l});
      
    end

end


%% (6) Determine optimal lambda parameter per participant
% --------------------------------------------------------

% load results from section above if not run
% load(generic_decoding_model.mat) 


% mean r / error within ROI
generic_r = squeeze(mean(CV_generic.r));
generic_err = squeeze(mean(CV_generic.err));

% Visual inspection of lambda trace
f3=figure()
subplot(1,2,1)
errorbar(1:nlambdas,generic_r,std(CV_generic.r)/sqrt(length(participants)),'linewidth',2)
set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
title('Cross-validated reconstruction accuracy')
xlabel('Regularization (1\times10^\lambda)')
ylabel('Pearsons r')
axis square

subplot(1,2,2)
errorbar(1:nlambdas,generic_err,std(CV_generic.err)/sqrt(length(participants)),'linewidth',2)
set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
title('Cross-validated mean squared error')
xlabel('Regularization (1\times10^\lambda)')
ylabel('Mean squared error (MSE)')
axis square

% only one optimal lambda value 
saveas(gcf, [datapath '/Decoder_s/lambda_generic.png']);

%%
% here, we select per participant the lambda parameter that maximizes the
% correlation between original and reconstructed onset envelope 
[max_r, idx_max] = max(CV_generic.r');
lambda_opt = 10.^(lambda_exp(idx_max));

% Train with optimal lambda value for each participants
for p = 1:length(participants)
    
    clear model fwd_model
    % train model on all data
    model = mTRFtrain(STIM{p},RESP{p},fs_ds,direction,tmin,tmax,lambda_opt(p));
    
    % forward-transform
    fwd_model = mTRFtransform(model, RESP{p});
              
    % store models from all participants in one structure
    MODELS_OPT(p,:,:) = model.w;
    CONST_OPT(p) = model.b;
    
    % forward-transformed model
    FWD_MODELS_OPT(p,:,:,:) = fwd_model.w;
        
    if p==1
         generic_model = model;
    end

end
    
   
% Average models per lambda across n-1 participants, test on left out participant
for p =1:length(participants)
    
    clear models consts
    % remove current participant       
    models = MODELS_OPT;
    models(p,:,:) = [];
     
    consts = CONST_OPT;
    consts(p) = [];
    
    % mean across models        
    generic_model.w = squeeze(mean(models(:,:,:)));
    generic_model.b = squeeze(mean(consts));
        
    % test model using left out data segment
    RECON_OPT{p} = mTRFpredict(STIM{p},RESP{p},generic_model);

    % compare prediction to actual EEG responses
    [STATS.r(p),STATS.err(p)] = mTRFevaluate(STIM{p}, RECON_OPT{p});
    
    save([datapath '/Decoder_s/Decoder_generic_vp' participants{p} '.mat'],'RECON_OPT','generic_model','STATS', 'lambda_opt', 'FWD_MODELS_OPT');

end


%% (7) Plot Temporal Response Function (TRF) for example participant
% --------------------------------------------------------------------

% exemplary fronto-central channel
chan=11; % FC6 
p = 1;  % show results for the same examplary participant '1' as above 

% Plot EEG, Stimulus, TRF, Reconstruction
figure
subplot(4,1,1)
plot((1:length(STIM{p}))/fs_ds,STIM{p});
title('Stimulus [s]')

subplot(4,1,2)
plot((1:length(RESP{p}))/fs_ds,RESP{p}(:,chan))
title('EEG [s]')

subplot(4,1,3)
plot(fwd_model.t,squeeze(FWD_MODELS_OPT(p,:,:,chan)))
xlim([0, 800])
title('Forward-transformed Temporal Response Function [ms]')

subplot(4,1,4)
plot((1:length(RECON_OPT{p}))/fs_ds,RECON_OPT{p})
title('Predicted EEG [s]')

%% (8) Compare prediction accuracy for individual and generic model optimization
% ------------------------------------------------------------------------------

% Plot prediction accuracy
f5=figure()
bar(1,stats.r)
hold on
bar(2,squeeze(STATS.r(p)))
hold off
set(gca,'xtick',1:2,'xticklabel',{'Individual','Generic'})
axis square
title('Model Performance')
xlabel('Dataset')
ylabel('Prediction accuracy [r]')
