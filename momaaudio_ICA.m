%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Preprocessing of EEG data for mTRF analysis for tutorial paper
%
%
%  Sarah J (May 2021)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define paths etc.
clear all
clc

restoredefaultpath

addpath '/Users/Win10/Documents/MATLAB/fieldtrip-20221103/';
addpath '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/scripts/';
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata';

ft_defaults
participants = {'1'};  

session = '1';


elecsEEG = { 'Fp1','Fp2','F9','F7','F3','Fz','F4','F8','F10','FC5','FC6','T7','C3','Cz','C4','T8',...
'TP9','CP5','CP6','TP10','P7','P3','PZ','P4','P8','O1','O2'};

    
    
    
%%
%%%%%%%%%%%%%%%%%%%%%%
% STEP 1: Read data %%
%%%%%%%%%%%%%%%%%%%%%%

for s=1:length(participants)
    clc
    display(['WORKING ON SUBJECT ' num2str(participants{s})])
    
  
        infile= [datapath '/moma-audio_vp' num2str(participants{s}) '_n' session '.eeg'];
        outfile=[datapath '/preprocessed/Segments_vp' num2str(participants{s}) '_n' session '.mat'];
    


    cfg=[];
    cfg.channel='all';
    cfg.reref='yes';         
    cfg.refchannel = elecsEEG; 
    cfg.detrend='yes';      % remove linear trend from data 
    cfg.demean='yes';       % apply baseline correction 
    cfg.lpfilter='yes';     % low pass and highpass filter
    cfg.lpfreq=40;
    cfg.hpfilter='yes';
    cfg.hpfreq=1;
    cfg.dataset=infile;
    continuous=ft_preprocessing(cfg);

    % segment data in 1-second snippets
    cfg=[];
    cfg.length = 1;
    cfg.overlap = 0; 
    preproc=ft_redefinetrial(cfg,continuous);
    
  
    
 
    save(outfile, 'preproc')
    
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%STEP 2: Automatic artifact rejection %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for s=1:length(participants)
    
    clc
    display(['WORKING ON SUBJECT ' num2str(participants{s})])
    
    infile=[datapath '/preprocessed/Segments_vp' num2str(participants{s}) '_n' session '.mat'];
    outfile=[datapath '/preprocessed/Cleaned_vp' num2str(participants{s}) '_n' session '.mat'];
    

       
    load(infile);
    
    cfg = [];
    cfg.artfctdef.win.channel = elecsEEG;
    cfg.artfctdef.win.rejlimit = 100; % what is the cut-off limit for std
    cfg.artfctdef.win.window = 100; % length of sliding window IN SAMPLING POINTS!!!
    [cfg,artifact1] = ft_artifact_multibaby(cfg,preproc);
    
    % Plotting
    cfg.continous = 'no';
    cfg.viewmode = 'vertical';
    cfg. ylim = [-25 25]; % scaling for the plotting of the data
    cfg = ft_databrowser(cfg,preproc); 
    
    cfg.artfctdef.reject  = 'complete';
    cleaned = ft_rejectartifact(cfg,preproc);
    
    save(outfile, 'cleaned')

end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: Perform ICA %%%
%%%%%%%%%%%%%%%%%%%%%%%%



for s=1:length(participants)
    
    clc
    display(['WORKING ON SUBJECT ' num2str(participants{s})])
    
    infile=[datapath '/preprocessed/Cleaned_vp' num2str(participants{s}) '_n' session '.mat'];
    outfile=[datapath '/preprocessed/Components_vp' num2str(participants{s}) '_n' session '.mat'];
    
    load(infile);
    
    % Determines maximum number of components (n)
    X = cat(3,cleaned.trial{:});
    X = reshape(X,[size(X,1) size(X,2)*size(X,3)])';
    [~, SVs, n, v] = svd_dim_reduction(X,'variance',0.95);
    
    cfg=[];
    cfg.channel=elecsEEG;
    cfg.ncomponent = n;
    cfg.method = 'fastica';
    comp=ft_componentanalysis(cfg, cleaned);

    
    save(outfile, 'comp')

end


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 4: Reject bad components from ICA %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for s=1:length(participants)

    clc
    display(['WORKING ON SUBJECT ' num2str(participants{s})])
    
    infile=[datapath '/preprocessed/Components_vp' num2str(participants{s}) '_n' session '.mat'];
    outfile=[datapath '/preprocessed/BadComponents_vp' num2str(participants{s}) '_n' session '.mat'];
    
    
    load(infile);

    % component rejection
    comp = ft_ica_powerspec(comp);
    
    cfg=[];
    cfg.freqmax=80;
    cfg.component='all';
    cfg.layout = 'infant.lay';
    cfg.xspacing = 'log';
    ft_componentbrowser_afft_new(cfg, comp)

    % specify component you want to reject
    clear bad components
    clc
    pause;
    bad_components=input('Components to reject: '); % e.g. [2 5 8]
% 
     save(outfile, 'bad_components')
    
end

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 5: Read and filter continuous raw data, remove bad ICA components %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% soundonset-trigger: 100; soundoffset-trigger: 101

% filter parameters
LP_Nfact = 8; % Factor of Filter Order for Low-Pass
HP_Nfact = 2; % Factor of Filter Order for High-Pass
FS_EEG = 500; % Sampling Rate
LP_EEG = 10;  % Low-Pass Cut off
HP_EEG = 1;   % High-Pass Cut off



for s=1:length(participants)
    
    clc
    display(['WORKING ON SUBJECT ' num2str(participants{s})])
    
    infile1=[datapath '/moma-audio_vp' num2str(participants{s}) '_n' session '.eeg'];
    infile2=[datapath '/preprocessed/Components_vp' num2str(participants{s}) '_n' session '.mat'];
    artifacts =[datapath '/preprocessed/Cleaned_vp' num2str(participants{s}) '_n' session '.mat'];
    infile3=[datapath '/preprocessed/BadComponents_vp' num2str(participants{s}) '_n' session '.mat'];
    outfile=[datapath '/preprocessed/PostICA' num2str(participants{s}) '_n' session  '.mat'];
        
    load(infile2);
    load(infile3);
    load(artifacts);   
 
    cfg=[];
    cfg.channel='all';
    cfg.reref='yes';
    cfg.refchannel = elecsEEG;
    cfg.detrend='yes';
    cfg.demean='yes';
    cfg.dataset=infile1;
    continuous=ft_preprocessing(cfg);
    
    % construct and apply filter
    % Low-Pass Coefficients
    b_lp = fir1(fix(LP_Nfact*FS_EEG/LP_EEG),LP_EEG/(FS_EEG/2));

    [H_lp,W] = freqz(b_lp,1,3*FS_EEG);

    % High-Pass Coefficients
    b_hp = fir1(fix(HP_Nfact*FS_EEG/HP_EEG),HP_EEG/(FS_EEG/2),'high');
    [H_hp,W] = freqz(b_hp,1,3*FS_EEG);
    
    filtered_data = continuous;
    for e = 1:length(elecsEEG)
        filtered_data.trial{1}(e,:) = filtfilt(b_hp,1,filtfilt(b_lp,1,continuous.trial{1}(e,:)'));
    end
    
    % reject ICA components
    cfg=[];
    cfg.component=bad_components;
    cfg.demean='no';
    preproc=ft_rejectcomponent(cfg,comp,filtered_data);
    % preproc = filtered_data;
    
    % segment data in 1-second snippets
    cfg=[];
    cfg.length = 1;
    cfg.overlap = 0;
    preproc=ft_redefinetrial(cfg,preproc);
    
  
    
    % take rejections from the very first step and exclude those
    cfg = [];
    cfg.artfctdef.win = cleaned.cfg.artfctdef.win;
    cfg.artfctdef.reject  = 'complete';
    preproc = ft_rejectartifact(cfg,preproc);

     % just included for revision
    cfg = [];
    cfg.continous = 'no';
    cfg.viewmode = 'vertical';
    cfg. ylim = [-25 25]; % scaling for the plotting of the data
    cfg = ft_databrowser(cfg,preproc); 
    % end of the "just included"
    
    % to get info when video started
     
    cfg=[];
    cfg.dataset=infile1;
    cfg.trialdef.eventtype = 'Stimulus';
    cfg.trialdef.eventvalue = {'S100';'S101'}; % Sound start und ende 
    cfg      = ft_definetrial(cfg);
    
    preproc.soundonset=cfg.trl(1);
    
    preproc.soundoffset=cfg.trl(2);
    
    % to get info when attention-getter occured, werden rausgenommen 
    cfg=[];
    cfg.dataset=infile1;
    cfg.trialdef.eventtype = 'Stimulus';
    cfg.trialdef.eventvalue = {'S 51';'S 52';'S 53';'S 54';'S 55';'S 56';'S 57';'S 58';'S 59';'S 71';'S 72';'S 73';'S 74';'S 75';'S 76';'S 77';'S 78';'S 79'};
    cfg      = ft_definetrial(cfg);
    
    
    x2=1;
    for x1 = 1:length(cfg.trl)/2
       attention(x1,1) = cfg.trl(x2);
       attention(x1,2) = cfg.trl(x2+1);
       x2=x2+2;
    end
    
    preproc.attentiongetter = attention;
    
    save(outfile, 'preproc')
    
end

