%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
%   Script aligns physical stimulus parameters  (in this case extracted sound) 
%   with cleaned EEG-data. In particular, the EEG-recording
%   starts before the sound, continues after the sound, and is missing for
%   chunks where data was noisy. Hence, we need to select on an individual
%   basis only those parts of the sound parameters during which EEG data
%   is available.
%   To do so, the fieldtrip structure gets a couple new entries.
%   preproc.resampled       contains the start and end point of the EEG epochs
%                           in sampling points shifted relative to the sound 
%                           onset, so that a sampling point of zero corresponds
%                           to the onset of the sound.
%   preproc.soundtrials     contains all eeg epochs that occured during the
%                           sound
%   preproc.soundsamples    contains the begin and end sample of all eeg
%                           epochs during the sound (sampling points are
%                           relative to the sound onset)
%   preproc.soundparameters contains soundparameters for the epochs included in
%                           preproc.soundtrials (i.e., that occured during
%                           the sound)
%
%   Sarah J, May 2021
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
parameterpath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/soundfiles/converted/';
datapath = '/Users/Win10/Documents/BScCoSci/Projects/EEG-EXP/MOMA-audio/rawdata/';

sound = [parameterpath '/extracted_feature.mat'];

participants = {'1'}

session = '1';

%%

for p = 1:length(participants)
    
    % load audio-feature
    sound = [parameterpath '/extracted_feature' participants{p} '.mat'];
    load(sound);
   
    % load EEG data
    eegdata = [datapath 'preprocessed/PostICA' num2str(participants{p}) '_n' session '.mat'];
    load(eegdata);
    % create new structure that is shifed by sampling point at which sound
    % occured so that 0 corresponds to the onset of the sound
    preproc.resampled = preproc.sampleinfo-preproc.soundonset;
    preproc.attention_rs = preproc.attentiongetter-preproc.soundonset;
    
    % these are the stimulus parameters we want to use as input; in this
    % case, it's only audio, but in principle we could add other parameters
    % here with values_combined = [parameterX parameterY parameterZ];
    values_combined = audio;
    
    % since the audiofile is presented in a loop during the experiment, we
    % need to append it as often as is was presented
    while length(values_combined)< ((preproc.soundoffset - preproc.soundonset)-length(audio))
        values_combined = [values_combined; audio];
    end
    
    % append the last repetion of the soundfile (which is typically only
    % part of the soundfile, since the experiment ended somewhere in the
    % middle)
    values_combined = [values_combined; audio(1:((preproc.soundoffset - preproc.soundonset)-length(values_combined)))];
    
    soundcounter = 1;
    % get all epochs that overlap with sound
    for i=1:length(preproc.resampled)
        
        % if the epoch in question occured during the video, copy the data
        % and the start and end point to videotrials and videosamples,
        % respectively
        if preproc.resampled(i,2)>=0 && preproc.resampled(i,1)<=length(values_combined)
            preproc.soundtrials{soundcounter} = preproc.trial{i};
            preproc.soundsamples(soundcounter,:) = preproc.resampled(i,:);
            soundcounter = soundcounter+1;  
        end             
    end
    
    
        % special treatment of first and last epoch, that might only
        % partially overlap with sound

            % for first epoch
            if preproc.soundsamples(1,1)<0
               preproc.soundtrials{1}= preproc.soundtrials{1}(:,length(preproc.soundtrials{1})-preproc.soundsamples(1,2)+1:length(preproc.soundtrials{1}));
               preproc.soundsamples(1,1) = 1;
            end
        
            % for last epoch
            if preproc.soundsamples(end,2)>length(values_combined)
               preproc.soundtrials{end}= preproc.soundtrials{end}(:,1:length(preproc.soundtrials{end})-(preproc.soundsamples(end,2)-length(values_combined)));
               preproc.soundsamples(end,2) = length(values_combined);
            end
            
            
        % special part for removing all those parts in which an attentiongetter was
        % played on top of the actual soundfile
        attentioncounter = 1;
        cleancounter = 1;
        cleaned_samples = [];
        cleaned_trials = {};

        for s = 1:length(preproc.soundsamples)

           % if a soundsample overlaps with an attentiongetter, i.e. if sampleonset (s,1) < attentiongetteroffset (attentioncounter,2) AND sampleoffset (s,2)
           % > attentiongetteronset (attentioncounter,1), do nothing but check
           % whether it's the end of the attentiongetter
           if preproc.soundsamples(s,1)<preproc.attention_rs(attentioncounter,2) && preproc.soundsamples(s,2)>preproc.attention_rs(attentioncounter,1)
               % "end of attentiongetter" is reached if the next sampleonset (s+1) is after the offset of the current attentiongetter. If so, move on
               % to the next attentiongetter by increasing the counter by 1
               if preproc.soundsamples(s+1,1)>preproc.attention_rs(attentioncounter,2) && attentioncounter < length(preproc.attention_rs)
                   attentioncounter = attentioncounter+1;
               end

           % else the soundsample does not overlap with an attentiongetter and is kept    
           else
               cleaned_samples(cleancounter,:) = preproc.soundsamples(s,:);
               cleaned_trials{cleancounter} = preproc.soundtrials{s};
               cleancounter = cleancounter+1;
           end
        end

        % replace previous entries with new structure without the
        % attentiongettersegments
        preproc.soundsamples = cleaned_samples;
        preproc.soundtrials = cleaned_trials;

        % end of the special part          
                        
            % add entry for audioparameters and fill them with the physical
            % stimulus parameters that overlap with the epochs just
            % selected
            for i=1:length(preproc.soundsamples)
               preproc.soundparameters{i} = values_combined(preproc.soundsamples(i,1):preproc.soundsamples(i,2),:);
            end
            
            save([datapath 'aligned/PostICAplusParameters' num2str(participants{p}) '_n' session '.mat'],'preproc');
    
end