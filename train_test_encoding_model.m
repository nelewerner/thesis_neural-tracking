%%TRAIN_TEST_ENCODING_MODEL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Training, tesing and visualization of encoding model using the
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

%% (1) Set parameters
% ----------------------

fs = 500; % EEG Sampling Rate

tmin = -200;  % start of time window used for training
tmax = 800;   % end of time window used for training
              % NOTE: when optimizing the model for predictive performance 
              % time lags should be restricted to periods capturing the 
              % strongest signal in the temporal response function
             
stim_param = 1;   % which stimulus parameter, here we only model the 
                  % auditory onset envelope as stimulus feature
                  
lambda_exp = -7:7; % range of hyperparameters
nlambdas = length(lambda_exp);
                           
direction = 1;     % direction of causality (forward=1, backward=-1) 

ROI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];   % Auditory ROI: F3, FZ, F4, FC5, FC6

% %% (2) Training and testing using data splits from the same single-subject dataset
% % --------------------------------------------------------------------------------

% mother - participants without noise
%participants = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger - participants without noise
%participants = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};


for p=1:length(participants)
    % load input data containing EEG responses and auditory stimulus features    
    load([datapath '/aligned/Inputdata' participants{p} '.mat']) 
    
    % normalize EEG responses (Important: time has to be the first dimension to preserve
    % relative differences between the individual EEG channels)
    resp = zscore(eegdata);  
    
    % stimulus feature for modelling (here, we only have one auditory feature)
    stim = stimulusdata(:,stim_param);
    
    % split data into k folds for nested crossvalidation
    k = 5; % number of folds
    fold_size = ceil(size(stim,1)/k); % ceil: rounds to nearest integers
    
    % Set aside first fold as validation set for final model evaluation that will
    % not be part of hyperparameter optimization, first fold for testing  
    idx = 1:min(fold_size*1,size(stim,1));
    stimtest = stim(idx,:);
    resptest = resp(idx,:);
    
    % all other folds are training sets used to optimize the hyperparameter
    stimtrain = cell(k-1,1);
    resptrain = cell(k-1,1);
    
    % data for model training
    for i = 1:k-1
        idx = fold_size*(i-1)+1:min(fold_size*i,size(stim,1));
        stimtrain{i,1} = stim(idx,:);
        resptrain{i,1} = resp(idx,:);
    end
    
    % Leave-one-out cross-validation to determine optimal lambda parameter
    % number of folds = number of instances in stimtrain 
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
    
            % train model on remaining data segments        
            MODEL(fold, l) = mTRFtrain(strain,rtrain,fs,direction,tmin,tmax,lambda);
    
            % test model using left out validation data segment
            PRED{fold,l} = mTRFpredict(sval,rval,MODEL(fold,l));
    
            % compare prediction to actual EEG responses
            [CV.r(fold,l,:),CV.err(fold,l,:)] = mTRFevaluate(rval, PRED{fold,l});
        end 
    
    end 

    %% (3) Determine optimal lambda parameter and test model on left-out data
    % ----------------------------------------------------------------------
    
    % assess prediction accuracy (Pearson's R) within the region of interest
    % (one may alternatively choose to evaluate the prediction accuracy more
    % globally using all channels)
    individ_r = squeeze(mean(CV.r(:,:,ROI),3));
    individ_err = squeeze(mean(CV.err(:,:,ROI),3));
    
    % Visual inspection of lambda trace
    f1=figure();
    subplot(1,2,1);
    errorbar(1:nlambdas,mean(individ_r),std(individ_r)/sqrt(k-1),'linewidth',2);
    set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1]);
    title('Cross-validated prediction accuracy [r]');
    xlabel('Regularization (1\times10^\lambda)');
    ylabel('Pearsons r');
    ylim([0 0.025]);
    axis square
    
    subplot(1,2,2);
    errorbar(1:nlambdas,mean(individ_err),std(individ_err)/sqrt(k-1),'linewidth',2);
    set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1]);
    title('Cross-validated mean squared error');
    xlabel('Regularization (1\times10^\lambda)');
    ylabel('Mean squared error (MSE)');
    ylim([1 1.1]);
    axis square
    
    
    % here, we select the lambda parameter that 
    % maximizes the correlation between predicted and recorded EEG 
    % within the region of interest 
    [max_r, idx_max] = max(squeeze(mean(mean(CV.r(:,:,ROI),3))));
    % define optimal lambda 
    lambda_opt = 10.^mean(lambda_exp(idx_max));
    
    % fit model with optimal lambda value on all of the training folds
    model = mTRFtrain(stimtrain,resptrain,fs,direction,tmin,tmax,lambda_opt);
    
    % test model on left out test data set
    [pred,stats] = mTRFpredict(stimtest,resptest,model);

    
    save([datapath '/MTRF_s/trf_individual_vp' participants{p} '.mat'],'pred','model','stats', 'lambda_opt');
    
    saveas(gcf, [datapath '/MTRF_s/lambda_individual_vp' participants{p} '.png']);

    %% (4) Plot Temporal Response Function (TRF)
    
    % exemplary fronto-central channel
    chan=11; % FC6 
    
    % Plot TRF (for one representative subject)
    f2=figure();
    subplot(4,1,1);
    plot((1:length(stimtest))/fs,stimtest);
    title('Stimulus [s]');
    
    subplot(4,1,2);
    plot((1:length(resptest))/fs,resptest(:,chan));
    title('EEG [s]');
    
    subplot(4,1,3);
    plot(model.t,model.w(1,:,chan)); 
    title('Temporal Response Function [ms]');
    
    subplot(4,1,4);
    plot((1:length(pred))/fs,pred(:,chan));
    title('Predicted EEG [s]');

    saveas(gcf, [datapath '/MTRF_s/trf_individual_vp' participants{p} '.png']);
end 

%% (5) Training and testing using a generic (subject-independent model)
% --------------------------------------------------------------------

% Note: you can alternatively to skip to the next section and load the 
% results of the cross-validation loop for final model fitting and testing 


% mother - participants without noise
%participants = {'2_n2','5_n1','7_n1','10_n2','16_n2', '17_n1', '22_n2', '23_n2', '28_n1','30_n1', '31_n2', '34_n2', '41_n2','43_n1', '46_n1'};

% stranger - participants without noise
%participants = {'1_n2', '13_n2', '16_n1', '17_n2', '19_n1', '22_n1', '23_n1', '28_n2', '34_n1', '35_n1', '37_n1', '43_n2','45_n1'};


% Train model for each participant using all data, afterwards before
% model training is compared with model testing, one participant is
% excluded 
for p = 1:length(participants)
    
    % load input data containing EEG responses and auditory stimulus features    
    load([datapath 'aligned/Inputdata' participants{p} '.mat']) 

    % normalize EEG responses (Important: time has to be the first dimension to preserve
    % relative differences between the individual EEG channels)
    RESP{p} = zscore(eegdata);  

    % stimulus feature for modelling (here, we only have one auditory feature)
    STIM{p} = stimulusdata(:,stim_param);

    % For each lambda value fit model to all data per participant 
    for l = 1:length(lambda_exp)
        
        clear model
        % set hyperparameter lambda
        lambda = 10.^lambda_exp(l); 

        % train model on all data, each lambda value 
        model = mTRFtrain(STIM{p},RESP{p},fs,direction,tmin,tmax,lambda);        
              
        % store models from all participants in one structure
        MODELS(p,l,:,:,:) = model.w;
        CONST(p,l,:) = model.b;

    end
    
    % keep model structure for creating generic models below 
    if p==1
         generic_model = model;
    end
end

% Average models per lambda across n-1 participants, test on left out participant
for p =1:length(participants)
    
    clear models consts
    % remove current participant by setting p = empty      
     models = MODELS;
     models(p,:,:,:,:) = [];
     
     consts = CONST;
     consts(p,:,:) = [];
    
    % for each lambda - test the optimal lambda 
    for l = 1:length(lambda_exp)
        
        % mean across models        
        generic_model.w(1,:,:) = squeeze(mean(models(:,l,:,:,:)));
        generic_model.b(1,:) = squeeze(mean(consts(:,l,:,:)));
        
        % test model using left out data segment
        PRED{p,l} = mTRFpredict(STIM{p},RESP{p},generic_model);

        % compare prediction to actual EEG responses
        [CV_generic.r(p,l,:),CV_generic.err(p, l,:)] = mTRFevaluate(RESP{p}, PRED{p,l});
      
    end

end


%% (6) Determine optimal lambda parameter across participants 
% --------------------------------------------------------------------

% load results from section above if not run
% load(generic_encoding_model.mat) 

% mean r / error within ROI
generic_r = squeeze(mean(CV_generic.r(:,:,ROI),3));
generic_err = squeeze(mean(CV_generic.err(:,:,ROI),3));

% Visual inspection of lambda trace
f3=figure()
subplot(1,2,1)
errorbar(1:nlambdas,mean(generic_r),std(generic_r)/sqrt(length(participants)),'linewidth',2)
set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
title('Cross-validated prediction accuracy [r]')
xlabel('Regularization (1\times10^\lambda)')
ylabel('Pearsons r')
ylim([0 0.04])
axis square

subplot(1,2,2)
errorbar(1:nlambdas,mean(generic_err),std(generic_err)/sqrt(length(participants)),'linewidth',2)
set(gca,'xtick',1:nlambdas,'xticklabel',lambda_exp), xlim([0,nlambdas+1])
title('Cross-validated mean squared error')
xlabel('Regularization (1\times10^\lambda)')
ylabel('Mean squared error (MSE)')
axis square

% only one optimal lambda value 
saveas(gcf, [datapath '/MTRF_s/lambda_generic.png']);


%%
% here, we select across participant the lambda parameter that maximizes the
% correlation between predicted and recorded EEG within the region of interest 
[max_r, idx_max] = max(mean(CV_generic.r(:,:,ROI),3)');
lambda_opt = mean(10.^(lambda_exp(idx_max)));

% Train with optimal lambda value for all participants
for p = 1:length(participants)
    
    clear model
    % train model on all data
    model = mTRFtrain(STIM{p},RESP{p},fs,direction,tmin,tmax,lambda_opt);        
              
    % store models from all participants in one structure
    MODELS_OPT(p,:,:,:) = model.w;
    CONST_OPT(p,:) = model.b;
    
    if p==1
         generic_model = model;
    end

end
    
   
% Average models per lambda across n-1 participants, test on left out participant
for p =1:length(participants)
    
    clear models consts
    % remove current participant       
    models = MODELS_OPT;
    models(p,:,:,:) = [];
     
    consts = CONST_OPT;
    consts(p,:) = [];
    
    % mean across models        
    generic_model.w(1,:,:) = squeeze(mean(models(:,:,:,:)));
    generic_model.b(1,:) = squeeze(mean(consts(:,:)));
        
    % test model using left out data segment
    PRED_OPT{p} = mTRFpredict(STIM{p},RESP{p},generic_model);

    % compare prediction to actual EEG responses
    [STATS.r(p,:),STATS.err(p,:)] = mTRFevaluate(RESP{p}, PRED{p});
    
    save([datapath '/MTRF_s/trf_generic_vp' participants{p} '.mat'],'PRED_OPT','generic_model','STATS', 'lambda_opt');
end



%% (7) Plot Temporal Response Function (TRF) across subjects
% --------------------------------------------------------------------
% (4) Plot Temporal Response Function (TRF)
% not needed because same code already from line 176
% exemplary fronto-central channel
chan=11; % FC6 

% Plot TRF (for one representative subject)
f2=figure();
subplot(4,1,1);
plot((1:length(stimtest))/fs,stimtest);
title('Stimulus [s]');

subplot(4,1,2);
plot((1:length(resptest))/fs,resptest(:,chan));
title('EEG [s]');

subplot(4,1,3);
plot(model.t,model.w(1,:,chan)); 
title('Temporal Response Function [ms]');

subplot(4,1,4);
plot((1:length(pred))/fs,pred(:,chan));
title('Predicted EEG [s]');


% exemplary fronto-central channel
chan=11; % FC6 
%p = 1;  % show results for the same examplary participant '1' as above 

for p =1:length(participants)

    % Plot TRF (for the same participant as before)
    f4=figure()
    subplot(4,1,1)
    plot((1:length(STIM{p}))/fs,STIM{p});
    title('Stimulus [s]');
    
    subplot(4,1,2)
    plot((1:length(RESP{p}))/fs,RESP{p}(:,chan))
    title('EEG [s]');
    
    subplot(4,1,3)
    plot(model.t,squeeze(MODELS_OPT(p,1,:,chan))) 
    title('Temporal Response Function [ms]')
    
    subplot(4,1,4)
    plot((1:length(PRED_OPT{p}))/fs,PRED_OPT{p}(:,chan))
    title('Predicted EEG [s]');
    
    saveas(gcf, [datapath '/MTRF_s/trf_generic_vp' participants{p} '.png']);

end
%% (8) Compare prediction accuracy for individual and generic model optimization
% ------------------------------------------------------------------------------

% Plot prediction accuracy
f5=figure()
bar(1,squeeze(mean(stats.r(1,ROI))))
hold on
bar(2,squeeze(mean(STATS.r(p,ROI))))
hold off
set(gca,'xtick',1:2,'xticklabel',{'Individual','Generic'})
axis square
title('Model Performance')
xlabel('Dataset')
ylabel('Prediction accuracy [r]')

saveas(gcf, [datapath '/MTRF_s/prediction_accuracy.png']);

