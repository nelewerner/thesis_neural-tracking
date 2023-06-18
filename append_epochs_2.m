%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%
%   Sarah J, May 2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all

datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata';

%participants = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '14', '15', '16', '17', '18', '19', '21',  '22', '23', '24', '25', '26', '27', '28', '29', '31', '32', '33', '34', '35', '37', '38', '41', '42', '43', '45'};
participants = {'1'};

session = '1';

% set overall parameters
zerolength = 500; % how many 0's to be included
elecs = 28; % number of EEG channels
stimulusparam = 1; % number of stimulus parameters included
FS = 500; %samplingrate
clear eegdata stimulusdata

for p = 1:length(participants)
    
    % load data
    data = [datapath '/aligned/PostICAplusParameters' participants{p} '_n' session '.mat'];
    load(data);
    
    % insert first epoch to create data structure for EEG and stimulus
    % parameters
    eegdata = preproc.soundtrials{1}';
    stimulusdata = preproc.soundparameters{1};
    
    % loop through all epochs
    for i = 2:length(preproc.soundsamples)
        
        % if the current epoch follows directly after the previous one,
        % just append it to the structure created above
        if preproc.soundsamples(i,1)==preproc.soundsamples(i-1,1)+FS;
           eegdata = vertcat(eegdata,preproc.soundtrials{i}'); 
           stimulusdata = vertcat(stimulusdata,preproc.soundparameters{i});
        
        % if the current epoch does not directly follow after the previous 
        % one, first append some 0's and then append the current epoch 
        % vertcat = vertical concatenation 
        elseif preproc.soundsamples(i,1)~= preproc.soundsamples(i-1,1)+FS
           eegdata = vertcat(eegdata,zeros(zerolength,elecs)); 
           stimulusdata = vertcat(stimulusdata,zeros(zerolength,stimulusparam));
           eegdata = vertcat(eegdata,preproc.soundtrials{i}'); 
           stimulusdata = vertcat(stimulusdata,preproc.soundparameters{i});
        end
        
    end
    
    save([datapath '/aligned/Inputdata' participants{p} '_n' session '.mat'],'eegdata','stimulusdata');
end

