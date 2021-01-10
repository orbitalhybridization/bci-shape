%% Preprocessing Pipeline
% Author: Ian Jackson


%% Set up containers and statics
%chans = [34,49,19,32,47,45,44,43,41,39,23,9,37,21]; % 14 channels of interest
refs = [29,26]; % 2 refs
ICx_all{1,5} = 0; % grand matrix for Independent Components
EEG_all{1,5} = 0; % grand matrix for all EEG structs (all shapes)

%% Determine whether we're processing for classification or for figure generation
processing_type = questdlg('Running for classification or figure generation?', ...
                            'Choose your path', ...
                            'I wanna classify!','Figures please!','Figures please!');
%% If we're doing figures, do that
if processing_type == "Figures please!"
    %% Collect paths for baseline and each shape class
    old_dataset = questdlg('Run same dataset as in last preprocessing session?', ...
                        'Choose your path', ...
                        'Yes, use the same one','No, use a new one','No use a new one');

    if old_dataset == "No, use a new one"
        shapes = [];
        while (isempty(shapes))
        [shapes,path] = uigetfile('C:\Users\scalplab\Desktop\ian\sessions',...
           'Select Shape File(s)', ...
           'MultiSelect', 'on');
        end

        baseline = [];
        while (isempty(baseline))
        [baseline,path] = uigetfile('C:\Users\scalplab\Desktop\ian\sessions',...
           'Select Baseline File(s)', ...
           'MultiSelect', 'on');
        end
    end

    %% Prompt for Preprocessing Parameters
    id_prompt = 'Enter participant name/identifier:';
    heatmaps_prompt = 'Generate Heatmaps';
    topo_prompt = 'Generate PSD Topographies';
    dlgtitle = 'Processing Parameters';
    dims = [1 35];
    definput = {'Kenpachi','all','true','true'};
    answer = inputdlg({id_prompt,heatmaps_prompt,topo_prompt},dlgtitle,dims,definput);
    identifier = answer{1};
    heatmaps = answer{2};
    topographies = answer{3};
    %% Main Preprocessing for Filtering
    % Run filtering -> Artefact rejection for shapes
    for i = 1:5 % Loop through shapes and save to grand matrix
        EEG = pop_loadbva(path+string(shapes(i)));
        EEG.setname = "eeg_data_all";

        % Filtering and Downsampling
        EEG = pop_eegfiltnew(EEG,0,83); %Low pass filter @ 83Hz
        EEG = pop_eegfiltnew( EEG, 49, 51, [], [1]); % Apply notch filter 50Hz
        EEG = pop_eegfiltnew( EEG, 59, 61, [], [1]); % Apply notch filter 60Hz

        % Downsample to 128Hz
        EEG = pop_resample(EEG,128);

        % Rereference to channels 29 and 26
        EEG = pop_reref(EEG,refs);

        %% First ICA and Artifact Rejection
        % Prune data with ICA (extended infomax)
        while (isempty(EEG.icasphere))
        EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
        end
        EEG = pop_selectcomps(EEG); % Display components to reject
        fig = uifigure;
        uiconfirm(fig,'Click when done selecting components.','Done',...
                    'CloseFcn','uiresume(fig)');
        uiwait(fig);

        %%
        %%EEG = pop_subcomp(EEG,[],0); %Reject selected components (default 1,2,3)
        %% 2nd ICA
        % Output: 63 waveforms
        %EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',chans);
        %Form ICx from weight and sphere matrices
        %EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
        ICx = EEG.icawinv*EEG.data(EEG.icachansind,:); %create 14 independent components
        % Set any components we want to reject to zero
        ICx(EEG.reject.gcompreject == 1,:) = zeros(size(ICx(EEG.reject.gcompreject==1,:))); 
        %ERROR condition: non-consecutive indices (e.g. 1 and 3 for
        %EEG.reject.gcompreject) cause imaginary numbers at runemd() step
        %%
        % use artifact rejected ICx to create a new X^
        EEG.data(EEG.icachansind,:) = EEG.icaweights*ICx;

        % If we're doing feature extraction with ICA, get components
        
        EEG_all{1,i} = EEG;
    end
    
    %% Filtering -> artifact rejection for baseline
    EEG = pop_loadbva(path+string(baseline));
    EEG.setname = "eeg_data_all";

    % Filtering and Downsampling
    EEG = pop_eegfiltnew(EEG,0,83); %Low pass filter @ 83Hz
    EEG = pop_eegfiltnew( EEG, 49, 51, [], [1]); % Apply notch filter 50Hz
    EEG = pop_eegfiltnew( EEG, 59, 61, [], [1]); % Apply notch filter 60Hz

    % Downsample to 128Hz
    EEG = pop_resample(EEG,128);

    % Rereference to channels 29 and 26
    EEG = pop_reref(EEG,refs);

    % First ICA and Artifact Rejection
    % Prune data with ICA (extended infomax)
    while (isempty(EEG.icasphere))
    EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
    end

    EEG = pop_selectcomps(EEG); % Display components to reject
    fig = uifigure;
    uiconfirm(fig,'Click when done selecting components.','Done',...
                'CloseFcn','uiresume(fig)');
    uiwait(fig);

    %%EEG = pop_subcomp(EEG,[],0); %Reject selected components (default 1,2,3)
    % 2nd ICA
    % Output: 63 waveforms
    %EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',chans);
    %Form ICx from weight and sphere matrices
    %EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
    ICx = EEG.icawinv*EEG.data(EEG.icachansind,:); %create 14 independent components
    % Set any components we want to reject to zero
    ICx(EEG.reject.gcompreject == 1,:) = zeros(size(ICx(EEG.reject.gcompreject==1,:))); 
    %ERROR condition: non-consecutive indices (e.g. 1 and 3 for
    %EEG.reject.gcompreject) cause imaginary numbers at runemd() step
    %
    % use artifact rejected ICx to create a new X^
    EEG.data(EEG.icachansind,:) = EEG.icaweights*ICx;

    % create new ICx_hat

    EEG_baseline = EEG;
    
    %%
    % Misc figure-generation from processing parameters
    if strcmp(topographies,'true')
        generate_topographies(EEG_all,EEG_baseline,identifier);
        fprintf("Topography Created");
    end

    if strcmp(heatmaps,'true')
       generate_heatmaps_ICs(training_power,identifier);
       fprintf("IC Heatmap Generated");
    end

    if strcmp(heatmaps,'true')
        generate_heatmaps(EEG_all,EEG_baseline,identifier);
        fprintf("PSD Heatmap Generated");
    end

%% Otherwise we're classifying
elseif processing_type == "I wanna classify!"
    %% Select files for baseline and shapes
    old_dataset = questdlg('Run same dataset as in last preprocessing session?', ...
                        'Choose your path', ...
                        'Yes, use the same one','No, use a new one','No use a new one');

    if old_dataset == "No, use a new one"
        shapes = [];
        while (isempty(shapes))
        [shapes,path] = uigetfile('C:\Users\scalplab\Desktop\ian\sessions',...
           'Select Shape File');
        end

        baseline = [];
        while (isempty(baseline))
        [baseline,path] = uigetfile('C:\Users\scalplab\Desktop\ian\sessions',...
           'Select Baseline File(s)', ...
           'MultiSelect', 'on');
        end
    end
    %% Preprocessing and epoch extraction
    answer = inputdlg({'Enter participant name/identifier:'},'Participant ID',[1 35],{'Kenpachi'});
    identifier = answer{1};

    %% Shapes Preprocessing
    EEG = pop_loadbva(path+string(shapes));
    EEG.setname = "eeg_data_all";
    EEG.comments = 'a';
    % Filtering and Downsampling
    EEG = pop_eegfiltnew(EEG,0,83); %Low pass filter @ 83Hz
    EEG = pop_eegfiltnew( EEG, 49, 51, [], [1]); % Apply notch filter 50Hz
    EEG = pop_eegfiltnew( EEG, 59, 61, [], [1]); % Apply notch filter 60Hz
    % Downsample to 128Hz
    EEG = pop_resample(EEG,128);
    % Rereference to channels 29 and 26
    EEG = pop_reref(EEG,refs);
    
    % save labels for later   
    targets{1,50} = 0;
    j = 1;
    for i = 1:100
        if strcmp(EEG.event(i).type,'S  2')
            targets{1,j} = 'Cube';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  3')
            targets{1,j} = 'Pyramid';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  4')
            targets{1,j} = 'Cylinder';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  5')
            targets{1,j} = 'Sphere';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  6')
            targets{1,j} = 'Cone';
            j=j+1;
        end
    end
    save(strcat(identifier,'_targets.m'),'targets');
%%
    % Extract epochs
    EEG = pop_epoch( EEG, {'S  2' 'S  3' 'S  4' 'S  5' 'S  6'}, [0.5 5]);
      
    % General ICA for Artifact Rejection
    EEG.icasphere = [];
    while (isempty(EEG.icasphere))
    EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
    end
    EEG = pop_selectcomps(EEG); % Display components to reject
    fig = uifigure;
    uiconfirm(fig,'Click when done selecting components.','Done',...
                'CloseFcn','uiresume(fig)');
    uiwait(fig);

    %ICx = EEG.icawinv*EEG.data(EEG.icachansind,:); %create 14 independent components
    ICx = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
    % Set any components we want to reject to zero
    ICx(EEG.reject.gcompreject == 1,:) = zeros(size(ICx(EEG.reject.gcompreject==1,:))); 
    % use artifact rejected ICx to create a new X^
    %EEG.data(EEG.icachansind,:) = EEG.icaweights*ICx;
    EEG.data(EEG.icachansind,:) = EEG.icawinv*ICx;

    % Save back into separate container
    saved_shape_data = EEG.data;

    % Per-trial ICA (2nd ICA)
    for i = 1:50 % we're gonna perform ICA on all trials just to get activations
        EEG.data = saved_shape_data(:,:,i);
        EEG.icasphere = [];
        while (isempty(EEG.icasphere))
            EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
        end
        %saved_shape_data(:,:,i) = EEG.icawinv*EEG.data;
        saved_shape_data(:,:,i) = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
    end
    
    % Put saved_data in a nice ordering
    %saved_shape_data = permute(saved_shape_data,[3 1 2]);
    
    %% Baseline Preprocessing
    EEG = pop_loadbva(path+string(baseline));
    EEG.setname = "eeg_data_all";
    EEG.comments = 'a';
    % Filtering and Downsampling
    EEG = pop_eegfiltnew(EEG,0,83); %Low pass filter @ 83Hz
    EEG = pop_eegfiltnew( EEG, 49, 51, [], [1]); % Apply notch filter 50Hz
    EEG = pop_eegfiltnew( EEG, 59, 61, [], [1]); % Apply notch filter 60Hz
    % Downsample to 128Hz
    EEG = pop_resample(EEG,128);
    % Rereference to channels 29 and 26
    EEG = pop_reref(EEG,refs);
    % Extract epochs
    EEG = pop_epoch( EEG, {'S  7'}, [0.5 2]);

    % General ICA for Artifact Rejection
    EEG.icasphere = [];
    while (isempty(EEG.icasphere))
    EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
    end
    EEG = pop_selectcomps(EEG); % Display components to reject
    fig = uifigure;
    uiconfirm(fig,'Click when done selecting components.','Done',...
                'CloseFcn','uiresume(fig)');
    uiwait(fig);

    %ICx = EEG.icawinv*EEG.data(EEG.icachansind,:); %create 14 independent components
    ICx = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
    % Set any components we want to reject to zero
    ICx(EEG.reject.gcompreject == 1,:) = zeros(size(ICx(EEG.reject.gcompreject==1,:))); 
    % use artifact rejected ICx to create a new X^
    %EEG.data(EEG.icachansind,:) = EEG.icaweights*ICx;
    EEG.data(EEG.icachansind,:) = EEG.icawinv*ICx;
    
    % Save into separate container
    saved_baseline_data = EEG.data; 

    % Per-trial ICA (2nd ICA)
    for i = 1:50 % we're gonna perform ICA on all trials just to get activations
        EEG.data = saved_baseline_data(:,:,i);
        EEG.icasphere = [];
        while (isempty(EEG.icasphere))
            EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
        end
        %saved_baseline_data(:,:,i) = EEG.icawinv*EEG.data;
        saved_baseline_data(:,:,i) = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
    end

    % Put saved_data in a nice ordering
    %saved_baseline_data = permute(saved_baseline_data,[3 1 2]);
    
    %% Then do PSD for each trial and normalize by baseline (if possible)
    %EEG = pop_loadset('filename','G00001_Ct_MMN.set','filepath','/data/mobi/Daisuke/p0100_icSelected_epoched/');

    freq_ranges = {[1 4],[5 8],[9 12],[13 30],[31 64]};
    %psd_data_norm = zeros([50 61 5]);

    %dataEpochRange     = 1:100;
    %baselineEpochRange = 201:300;
    
    dataEpochRange = 1:50;
    baselineEpochRange = 1:50;

    %saved_shape_data    = EEG.data(:,:,dataEpochRange);
    %saved_baseline_data = EEG.data(:,:,baselineEpochRange);

    meanPowerMicroV          = zeros(EEG.nbchan, 5, length(dataEpochRange));
    meanPowerMicroV_baseline = zeros(EEG.nbchan, 5, length(baselineEpochRange));

    % Obtain data-epoch PSD.
    for dataEpochIdx = 1:length(dataEpochRange) % For each data epoch.
        %current_data = squeeze(saved_shape_data(:,:,dataEpochIdx));
        current_data = squeeze(saved_shape_data(:,:,dataEpochIdx));

        [allElecPsd, freqBins] = spectopo(current_data, 0, 128, 'plot', 'off');

        if dataEpochIdx == 1
            freqRange1Idx = find(freqBins >= freq_ranges{1}(1) & freqBins <= freq_ranges{1}(2));
            freqRange2Idx = find(freqBins >= freq_ranges{2}(1) & freqBins <= freq_ranges{2}(2));
            freqRange3Idx = find(freqBins >= freq_ranges{3}(1) & freqBins <= freq_ranges{3}(2));
            freqRange4Idx = find(freqBins >= freq_ranges{4}(1) & freqBins <= freq_ranges{4}(2));
            freqRange5Idx = find(freqBins >= freq_ranges{5}(1) & freqBins <= freq_ranges{5}(2));
        end

        meanPowerMicroV(:,:,dataEpochIdx) = [mean(10.^((allElecPsd(:,freqRange1Idx))/10), 2) ...
                                             mean(10.^((allElecPsd(:,freqRange2Idx))/10),2) ...
                                             mean(10.^((allElecPsd(:,freqRange3Idx))/10),2) ...
                                             mean(10.^((allElecPsd(:,freqRange4Idx))/10),2) ...
                                             mean(10.^((allElecPsd(:,freqRange5Idx))/10),2)];
    end

    % Obtain baseline-epoch PSD.
    for baselineEpochIdx = 1:length(dataEpochRange) % For each baseline epoch.
        current_baseline_data = squeeze(saved_baseline_data(:,:,baselineEpochIdx));

        [allElecPsd, freqBins] = spectopo(current_baseline_data, 0, 128,'plot', 'off');

        if baselineEpochIdx == 1
            freqRange1Idx = find(freqBins >= freq_ranges{1}(1) & freqBins <= freq_ranges{1}(2));
            freqRange2Idx = find(freqBins >= freq_ranges{2}(1) & freqBins <= freq_ranges{2}(2));
            freqRange3Idx = find(freqBins >= freq_ranges{3}(1) & freqBins <= freq_ranges{3}(2));
            freqRange4Idx = find(freqBins >= freq_ranges{4}(1) & freqBins <= freq_ranges{4}(2));
            freqRange5Idx = find(freqBins >= freq_ranges{5}(1) & freqBins <= freq_ranges{5}(2));
        end

        % Calculate baseline?
        meanPowerMicroV_baseline(:,:,baselineEpochIdx) = [mean(10.^((allElecPsd(:,freqRange1Idx))/10), 2) ...
                                                          mean(10.^((allElecPsd(:,freqRange2Idx))/10), 2) ...
                                                          mean(10.^((allElecPsd(:,freqRange3Idx))/10), 2) ...
                                                          mean(10.^((allElecPsd(:,freqRange4Idx))/10), 2) ...
                                                          mean(10.^((allElecPsd(:,freqRange5Idx))/10), 2)];
    end

    % Normalize data-epoch PSD with baseline-epoch PSD.
    psd_data_norm = meanPowerMicroV./meanPowerMicroV_baseline;
    psd_data_norm = reshape(psd_data_norm,[50 61 5]);

    % save
    save(strcat(identifier,'_psd_data_normalized.m'),'psd_data_norm');
end

f = msgbox('Processing Completed');

%% Get PSD for all shapes
function [normalized_PSDs] = getPSD(all_data,baseline)
    % This example code compares PSD in uV^2/Hz rendered as scalp topography (setfile must be loaded.)
    % Set up statics
    freq_ranges = {[1 4],[5 8],[9 12],[13 30],[31 64]}; % this might not be the best way to set up ranges....
    baseline_PSDs{1,5} = 0; % cell array for baseline PSDs per freq. band
    
    % First calculate PSD for baseline to divide later
    for i = 1:5
        meanPowerMicroV_baseline = zeros(baseline.nbchan,1);
        for channelIdx = 1:baseline.nbchan
            [psdOutDb(channelIdx,:), freq] = spectopo(baseline.data(channelIdx, :), 0, baseline.srate, 'plot', 'off');
            lowerFreqIdx    = find(freq==freq_ranges{i}(1));
            higherFreqIdx   = find(freq==freq_ranges{i}(2));
            %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
            meanPowerMicroV_baseline(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
        end
        baseline_PSDs{1,i} = meanPowerMicroV_baseline; % save to cell array
    end
    
    % Then do PSD for each shape and generate heatmap
    normalized_PSDs = zeros([61 5]); %normalized PSDs for each freq_band
    meanPowerMicroV = zeros(all_data.nbchan,1);
    for i = 1:5 % for all frequency bands
        for channelIdx = 1:all_data.nbchan
            [psdOutDb(channelIdx,:), freq] = spectopo(all_data.data(channelIdx, :), 0, all_data.srate, 'plot', 'off');
            lowerFreqIdx    = find(freq==freq_ranges{i}(1));
            higherFreqIdx   = find(freq==freq_ranges{i}(2));
            %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
            meanPowerMicroV(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
        end
        meanPowerMicroV = meanPowerMicroV ./ baseline_PSDs{1,i}; % normalize by baseline
        normalized_PSDs(:,i) = squeeze(meanPowerMicroV);
    end
end
%% Close Box
function my_closereq(fig,selection)

    selection = uiconfirm(fig,'Close the figure window?',...
        'Confirmation');

    switch selection
        case 'OK'
            delete(fig)

        case 'Cancel'
            return
    end

end

%% Generate Heatmaps from ICs
function generate_heatmaps_ICs(all_data,identifier)
    labels = ["Cone","Cube","Cylinder","Pyramid","Sphere"];
    for o = 1:5 %go through all shapes and make heatmaps for this person
        subplot(1,5,o);
        xvalues = {"Delta","Theta","Alpha","Beta","Gamma"};
        yvalues{1,61} = 0;
        for i = 1:61 % make cell array of numbers in range 1:61
            yvalues{1,i} = int2str(i);
        end
        %yvalues = {"1","2","3","4","5","6","7","8","9","10","11","12","13","14"};
        %T = table(Comps,Delta,Theta,Alpha,Beta,Gamma);
        heatmap(xvalues,yvalues,squeeze(all_data(o,:,:)));
        title(labels(o));
    end
    fig_title = strcat(identifier,'_ICAheatmap.png');
    set(gcf, 'Position', get(0, 'Screensize'));    
    saveas(gcf,fig_title)
end

%% Generate heatmaps from power spectra
% From : https://sccn.ucsd.edu/wiki/Makoto's_useful_EEGLAB_code#How_to_extract_EEG_power_of_frequency_bands_.2806.2F06.2F2020_updated.29
function generate_heatmaps(all_data,baseline,identifier)
 % This example code compares PSD in uV^2/Hz rendered as scalp topography (setfile must be loaded.)
    % Set up statics
    freq_ranges = {[1 4],[5 8],[9 12],[13 30],[31 64]}; % this might not be the best way to set up ranges....
    freq_names = {"Delta","Theta","Alpha","Beta","Gamma"};
    labels = ["Cone","Cube","Cylinder","Pyramid","Sphere"];
    baseline_PSDs{1,5} = 0; % cell array for baseline PSDs per freq. band
    
    % First calculate PSD for baseline to divide later
    for i = 1:5
        meanPowerMicroV_baseline = zeros(baseline.nbchan,1);
        for channelIdx = 1:baseline.nbchan
            [psdOutDb(channelIdx,:), freq] = spectopo(baseline.data(channelIdx, :), 0, baseline.srate, 'plot', 'off');
            lowerFreqIdx    = find(freq==freq_ranges{i}(1));
            higherFreqIdx   = find(freq==freq_ranges{i}(2));
            %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
            meanPowerMicroV_baseline(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
        end
        baseline_PSDs{1,i} = meanPowerMicroV_baseline; % save to cell array
    end
    
    % Then do PSD for each shape and generate heatmap
    for j = 1:5 % j = shape index
        normalized_PSDs = zeros([61 5]); %normalized PSDs for each freq_band
        current_data = all_data{j};
        meanPowerMicroV = zeros(current_data.nbchan,1);
        for i = 1:5 % for all frequency bands
            for channelIdx = 1:current_data.nbchan
                [psdOutDb(channelIdx,:), freq] = spectopo(current_data.data(channelIdx, :), 0, current_data.srate, 'plot', 'off');
                lowerFreqIdx    = find(freq==freq_ranges{i}(1));
                higherFreqIdx   = find(freq==freq_ranges{i}(2));
                %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
                meanPowerMicroV(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
            end
            meanPowerMicroV = meanPowerMicroV ./ baseline_PSDs{1,i}; % normalize by baseline
            normalized_PSDs(:,i) = squeeze(meanPowerMicroV);
        end
        subplot(1,5,j);
        yvalues{1,61} = 0;
        for i = 1:61 % make cell array of numbers in range 1:61
            yvalues{1,i} = int2str(i);
        end
        %yvalues = {"1","2","3","4","5","6","7","8","9","10","11","12","13","14"};
        %T = table(Comps,Delta,Theta,Alpha,Beta,Gamma);
        heatmap(freq_names,yvalues,normalized_PSDs);
        title(labels(j));
    end
    
    fig_title = strcat(identifier,'_PSDheatmap.png');
    set(gcf, 'Position', get(0, 'Screensize'));    
    saveas(gcf,fig_title)
end

%% Generate Topographies
function generate_topographies(all_data,baseline,identifier) % From https://sccn.ucsd.edu/wiki/Makoto's_useful_EEGLAB_code#How_to_extract_EEG_power_of_frequency_bands_.2806.2F06.2F2020_updated.29
    % This example code compares PSD in uV^2/Hz rendered as scalp topography (setfile must be loaded.)
    % Set up statics
    freq_ranges = {[1 4],[5 8],[9 12],[13 30],[31 64]}; % this might not be the best way to set up ranges....
    freq_names = {"Delta","Theta","Alpha","Beta","Gamma"};
    labels = ["Cone","Cube","Cylinder","Pyramid","Sphere"];
    data_to_avg = zeros([5 5 61]);
    
    % First calculate PSD for baseline to divide later
    baseline_PSDs{1,5} = 0; % cell array for baseline PSDs per freq. band
    for i = 1:5
        meanPowerMicroV_baseline = zeros(baseline.nbchan,1);
        for channelIdx = 1:baseline.nbchan
            [psdOutDb(channelIdx,:), freq] = spectopo(baseline.data(channelIdx, :), 0, baseline.srate, 'plot', 'off');
            lowerFreqIdx    = find(freq==freq_ranges{i}(1));
            higherFreqIdx   = find(freq==freq_ranges{i}(2));
            %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
            meanPowerMicroV_baseline(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
        end
        baseline_PSDs{1,i} = meanPowerMicroV_baseline; % save to cell array
    end
    
    % Then do PSD for each shape
    p = 1;
    for j = 1:5 % j = shape index
        current_data = all_data{j};
        meanPowerMicroV = zeros(current_data.nbchan,1);
        for i = 1:5 % for all frequency bands
            for channelIdx = 1:current_data.nbchan
                [psdOutDb(channelIdx,:), freq] = spectopo(current_data.data(channelIdx, :), 0, current_data.srate, 'plot', 'off');
                lowerFreqIdx    = find(freq==freq_ranges{i}(1));
                higherFreqIdx   = find(freq==freq_ranges{i}(2));
                %meanPowerDb(channelIdx) = mean(psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx));
                meanPowerMicroV(channelIdx) = mean(10.^((psdOutDb(channelIdx, lowerFreqIdx:higherFreqIdx))/10), 2);
            end
            meanPowerMicroV = meanPowerMicroV ./ baseline_PSDs{1,i}; % normalize by baseline
            data_to_avg(j,i,:) = meanPowerMicroV;
            subplot(5,5,p)
            topoplot(meanPowerMicroV, current_data.chanlocs)
            title(freq_names{i} + " (" + labels(j) + ")")
            cbarHandle = colorbar;
            set(get(cbarHandle, 'title'), 'string', '(uV^2/Hz)')
            p = p + 1;
        end
    end
    fig_title = strcat(identifier,'_PSDtopography.png');
    set(gcf, 'Position', get(0, 'Screensize'));    
    saveas(gcf,fig_title)
    save('data_to_avg.m','data_to_avg');
end

%% MISC: Hilbert Huang Transform
%{

% Set up final matrix for power bands of each component
[allcomps] = zeros([5 14]);

% First step: Empirical Mode Decomposition
% 1. Forms a list of Intrinsic mode functions from decomposed signals
   % IMFs are functions that simplify the complicated signal into
   % wavelets with specific properties
[allmodes] = runemd(ICx_hat,'modes',6); % Empirical Mode Decomposition per component
[allmodes] = allmodes(:,2:6,:); %chop the first imf, which is the original data
%%
% Main Loop (Calculate power for 5 freq. bands of each component)
    %We need to extract a 6(number of modes) x size(data) matrix from each component to pass to hht
    current_component_imfs = squeeze(allmodes(5,:,:));
    current_component_imfs = current_component_imfs'; %transpose
    
    % Second step: Hilbert Spectrum Analysis
    % This is used to calculate the power of five freq. bands:
    % First, we'll calculate the hilbert spectrum of the IMFs
    [hs,f,t,imfinsf,imfinse] = hht(current_component_imfs,128);
    
    % Then the marginal spectrum can be calculated by integrating the hilbert
    % spectrum over time
    %[A F] = mhs(imfinse,imfinsf,0,0.1,128);
    [MHSp,HMSpSmooth] = HMSmeb(imfinsf(:,1),imfinse(:,1),0.002,0.5);
    plot(MHSp(:,1),MHSp(:,2));
    %%
for i = 4:7
    % This will give us a vector for frequencies (F) and a vector that is the sum
    % of the contribution of each frequency component.
    
    % We'll next find the indices in F that correspond to each band:
    % Delta(1-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz),
    % and Gamma (30-64Hz).
    [delta_idx] = find(1<=F & F<4);
    [theta_idx] = find(4<=F & F<8);
    [alpha_idx] = find(8<=F & F<12);
    [beta_idx] = find(12<=F & F<30);
    [gamma_idx] = find(30<=F & F<64);
    
    % Next, we'll find avg power of frequency bands we're interested in
    delta_sum_avg = mean(A(delta_idx));
    theta_sum_avg = mean(A(theta_idx));
    alpha_sum_avg = mean(A(alpha_idx));
    beta_sum_avg = mean(A(beta_idx));
    gamma_sum_avg = mean(A(gamma_idx));

    % Finally, we'll save these means in the grand matrix for each component and
    % loop to the next one
    allcomps(:,i) = horzcat(delta_sum_avg,theta_sum_avg,alpha_sum_avg,beta_sum_avg,gamma_sum_avg);
end

%}

%% MISC: Feature Extraction of ICA Components
%{
%shapes_features = []; % grand matrix for features
%baseline_features = []; % grand matrix for baselines features
% In the end we can play with ICx_all and ICx_hat_baseline
%% Simple Feature Extraction for Classification Testing
[P F] = spec(ICx_hat); % get power spectra of ICA components
F = F.^(-1); %invert frequencies
 
    % We'll first find the indices in F that correspond to each band:
    % Delta(1-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz),
    % and Gamma (30-64Hz).
    [delta_idx] = find(1<=F & F<4);
    [theta_idx] = find(4<=F & F<8);
    [alpha_idx] = find(8<=F & F<12);
    [beta_idx] = find(12<=F & F<30);
    [gamma_idx] = find(30<=F & F<=64);
   
    % Finally, we'll save the power of the frequency bands in the grand
    % matrix
    allcomps = padcat(P(delta_idx),P(theta_idx),P(alpha_idx),P(beta_idx),P(gamma_idx));

baseline_features = allcomps;
%}
%{
%% Simple Feature Extraction for Classification Testing
[P F] = spec(ICx_hat); % get power spectra of ICA components
F = F.^(-1); %invert frequencies
 
    % We'll first find the indices in F that correspond to each band:
    % Delta(1-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz),
    % and Gamma (30-64Hz).
    [delta_idx] = find(1<=F & F<4);
    [theta_idx] = find(4<=F & F<8);
    [alpha_idx] = find(8<=F & F<12);
    [beta_idx] = find(12<=F & F<30);
    [gamma_idx] = find(30<=F & F<=64);
   
    % Finally, we'll save the power of the frequency bands in the grand
    % matrix
    allcomps = padcat(P(delta_idx),P(theta_idx),P(alpha_idx),P(beta_idx),P(gamma_idx));

if isempty(shapes_features)
    shapes_features = allcomps;
else
    %shapes_features = (padcat(shapes_features,allcomps));
    shapes_features = [shapes_features;allcomps];
end
end
shapes_features = shapes_features';
%}
%{
% Reshape grand matrix into 3-dim matrix with classes in first dim
reshaped_shapes = zeros([5 0 0]);
for j = 1:5
    index = 5*j - 4;
    reshaped_shapes(j,:,:) = shapes_features(index:index+4,:);
end

%% Feature Extraction (ICA)
if strcmp(feature_extraction_method,'ICA')
    %function feature_extract_from_ICA
    % split data into 70:20:10 for each shape
    training_ICs{1,5} = [0];
    test_ICs{1,5} = [0];
    rest_ICs{1,5} = [0];
    trial_size = size(ICx_all{1,1}{1},2);
    training_index = (cast(0.7 * trial_size,'uint64'));
    test_index = cast(0.2 * trial_size,'uint64') + training_index;
    
    % Chop baseline to the same length as experimental data
    ICx_baseline = ICx_baseline(:,1:trial_size);
    % Then divide to normalize for each shape
    for i = 1:5
        for j = 1:61 % do this for each component because, for some reason it changes to 0.00000
            ICx_all{1,i}{1}(:,j) = (ICx_all{1,i}{1}(:,j))./(ICx_baseline(:,j)); % divide each by the baseline
        end
    end
    
    % Then split up
    for i = 1:5
        training_ICs{1,i} = ICx_all{1,i}{1}(:,1:training_index);
        test_ICs{1,i} = ICx_all{1,i}{1}(:,training_index+1:test_index);
        rest_ICs{1,i} = ICx_all{1,i}{1}(:,test_index+1:trial_size);
    end
    
    % Get power spectra of each component for each shape for 5 frequency bands
    % For training
    training_power = zeros([5 61 5]);
    for i = 1:5 % Go through all shapes
        for j = 1:61 % go through each component and create features :D
            current_component = training_ICs{1,i}(j,:);
            [P F] = spectopo(current_component,0,EEG.srate);
            % We'll next find the indices in F that correspond to each band:
            % Delta(1-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz),
            % and Gamma (30-64Hz).
            [delta_idx] = find(F>1 & F<4);
            [theta_idx] = find(F>4 & F<8);
            [alpha_idx] = find(F>8 & F<13);
            [beta_idx] = find(F>13 & F<30);
            [gamma_idx] = find(F>30 & F<80);

            % Next, we'll find avg power of frequency bands we're interested in
            deltaPower = mean(10.^(P(delta_idx)/10));
            thetaPower = mean(10.^(P(theta_idx)/10));
            alphaPower = mean(10.^(P(alpha_idx)/10));
            betaPower = mean(10.^(P(beta_idx)/10));
            gammaPower = mean(10.^(P(gamma_idx)/10));

            % Finally, we'll save these means in the grand matrix for each component and
            % loop to the next one
            current_feature = horzcat(deltaPower,thetaPower,alphaPower,betaPower,gammaPower);
            training_power(i,j,:) = current_feature;
        end
    end
    % For test
    test_power = zeros([5 61 5]);
    for i = 1:5 % Go through all shapes
        for j = 1:61 % go through each component and create features :D
            current_component = test_ICs{1,i}(j,:);
            [P F] = spectopo(current_component,0,EEG.srate);
            F = F.^(-1); %invert frequencies
            % We'll next find the indices in F that correspond to each band:
            % Delta(1-4Hz), Theta (4-8Hz), Alpha (8-12Hz), Beta (12-30Hz),
            % and Gamma (30-64Hz).
            [delta_idx] = find(1<F & F<4);
            [theta_idx] = find(4<F & F<8);
            [alpha_idx] = find(8<F & F<12);
            [beta_idx] = find(12<F & F<30);
            [gamma_idx] = find(30<F & F<64);

            % Next, we'll find avg power of frequency bands we're interested in
            deltaPower = mean(10.^(P(delta_idx)/10));
            thetaPower = mean(10.^(P(theta_idx)/10));
            alphaPower = mean(10.^(P(alpha_idx)/10));
            betaPower = mean(10.^(P(beta_idx)/10));
            gammaPower = mean(10.^(P(gamma_idx)/10));


            % Finally, we'll save these means in the grand matrix for each component and
            % loop to the next one
            current_feature = horzcat(deltaPower,thetaPower,alphaPower,betaPower,gammaPower);
            test_power(i,j,:) = current_feature;
        end
    end
end
%}
%--------------------------------------%
%% MISC: Feature Plotting
%{
function fig = plot_features(~)
%% Plotting Features
freq_strings = {'\delta','\theta','\alpha','\beta','\gamma'};
figure(1);
n = 1;
for m = 1:25 % loop through all subplots
        subplot(5,5,m);
        if m < 6
            title(freq_strings(m),'FontSize',16); % set and if statement for the first freq.
        end
        if mod(m,5) == 0
            h = 5;
        else
            h = mod(m,5);
        end
        topoplot(all_shapes_normalized(n,h,:),EEG.chanlocs(chans));
        %caxis([0 1.5])
        if h == 5
           n = n + 1; 
        end
end
%}

%% MISC: Set Targets in Events Themselves
%{

%m = set_targetmarkers(EEG,{'S 2','S 3','S 4','S 5','S 6','S 7'})
m = set_targetmarkers(EEG,{'S 2','S 3','S 4','S 5','S 6'});
for i = 1:100
    for j = 1:6
        if strcmp(EEG.event(i).type(strlength(EEG.event(i).type)),m.parts{2}{j}(strlength(m.parts{2}{j})))
            EEG.event(i).target = m.parts{2}{j};
            EEG.event(i).type = m.parts{2}{j};
        end
    end
end
%}

%% MISC: Find Missed Targets and Accuracy
%{
% Here we'll find the missed targets using lastresults
% and the loaded EEG
%%
% Here we'll find the missed targets using lastresults
% and the loaded EEG
%%
preds = zeros(1,10);
targets{1,10} = 0;

% Get predictions, and targets
for i = 1:10
    preds(i) = argmax(lastresults.prediction{2}(i,:));
    targets{1,i} = str2double(EEG.event(i).target(strlength(EEG.event(i).target)));
end

% Translate predictions back to shape numbers
for i = 1:10
    preds(1,i) = preds(1,i)+1;
end

fprintf("\n\nMissed Targets\n")
disp("------------------")
missed = 0;
for i = 1:10
    if preds(i) ~= targets{1,i}
        fprintf('Target: %i',targets{1,i})
        fprintf(' Prediction: %i\n',preds(i))
        missed = missed + 1;
    end
end

fprintf("Prediction Accuracy (approx.): %f", (10-missed)/10);
%}

%% MISC: Extract Class Names from EEG.event
%{
targets{1,50} = 0;
j = 1;
for i = 1:100
        if strcmp(EEG.event(i).type,'S  2')
            targets{1,j} = 'Cube';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  3')
            targets{1,j} = 'Pyramid';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  4')
            targets{1,j} = 'Cylinder';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  5')
            targets{1,j} = 'Sphere';
            j=j+1;
        elseif strcmp(EEG.event(i).type,'S  6')
            targets{1,j} = 'Cone';
            j=j+1;
        end
end
%}

%% MISC: ICA Per-trial
%{
%% Data extraction
saved_data = EEG.data;

%% General ICA for Artifact Rejection
EEG.icasphere = [];
while (isempty(EEG.icasphere))
EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
end
EEG = pop_selectcomps(EEG); % Display components to reject
fig = uifigure;
uiconfirm(fig,'Click when done selecting components.','Done',...
            'CloseFcn','uiresume(fig)');
uiwait(fig);

%%EEG = pop_subcomp(EEG,[],0); %Reject selected components (default 1,2,3)
% Output: 63 waveforms
%EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',chans);
%Form ICx from weight and sphere matrices
%EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
ICx = EEG.icawinv*EEG.data(EEG.icachansind,:); %create 14 independent components
% Set any components we want to reject to zero
ICx(EEG.reject.gcompreject == 1,:) = zeros(size(ICx(EEG.reject.gcompreject==1,:))); 
%ERROR condition: non-consecutive indices (e.g. 1 and 3 for
%EEG.reject.gcompreject) cause imaginary numbers at runemd() step

% use artifact rejected ICx to create a new X^
EEG.data(EEG.icachansind,:) = EEG.icaweights*ICx;

% Per-trial ICA (2nd ICA)
for i = 1:50 % we're gonna perform ICA on all trials just to get activations
EEG.data = saved_data(:,:,i);
%% First ICA and Artifact Rejection
% Prune data with ICA (extended infomax)
EEG.icasphere = [];
while (isempty(EEG.icasphere))
EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',1:61);
end
saved_data(:,:,i) = EEG.icawinv*EEG.data;
end
saved_data = permute(saved_data,[3 1 2]);
%}