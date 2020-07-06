%% Collect paths (a workaround) so that we can collect comps per band per shape
path = ["C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_segmentation_cube.mat",
        "C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_Segmentation_baseline_cube.mat",
        "C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_segmentation_sphere.mat",
        "C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_Segmentation_baseline_sphere.mat",
        "C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_segmentation_pyramid.mat",
        "C:\Users\scalplab\Desktop\ian\testdata\session 2\export\shapetestplease_Segmentation_baseline_pyramid.mat"];
  
% Cube, Baseline = 1,2
% Sphere,Baseline = 3,4
% Pyramic, Baseline = 5,6

chans = [34,49,19,32,47,45,44,43,41,39,23,9,37,21]; % 14 channels of interest
refs = [29,26]; % 2 refs

%% Preprocessing
% Import exported .mat file
path = path(6); % dont' forget to change this and allshapesfeatures to the next shape index
EEG = pop_loadbva(path);
EEG.setname = "eeg_data_all";

EEG = pop_eegfiltnew(EEG,0,83); %Low pass filter @ 83Hz
EEG = pop_eegfiltnew( EEG, 49, 51, [], [1]); % Apply notch filter 50Hz
EEG = pop_eegfiltnew( EEG, 59, 61, [], [1]); % Apply notch filter 60Hz

% Downsample to 128Hz
EEG = pop_resample(EEG,128);

% Rereference to channels 29 and 26
EEG = pop_reref(EEG,refs);

%% First ICA and Artifact Rejection
% Prune data with ICA (extended infomax)
EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',chans);
%%
EEG = pop_selectcomps(EEG, 1:14 ); % Display components to reject
%%
EEG = pop_subcomp(EEG,[],0); %Reject selected components (default 1,2,3)
%% 2nd ICA
% Output: 63 waveforms
EEG = pop_runica(EEG,'extended',1,'stop',1e-07,'interrupt','on','chanind',chans);

%% Form ICx from weight and sphere matrices
EEG.icaact = (EEG.icaweights*EEG.icasphere)*EEG.data(EEG.icachansind,:);
%% Hilbert Huang Transform

% Set up final matrix for power bands of each component
[allcomps] = zeros([5 14]);

% First step: Empirical Mode Decomposition
% 1. Forms a list of Intrinsic mode functions from decomposed signals
   % IMFs are functions that simplify the complicated signal into
   % wavelets with specific properties
[allmodes] = runemd(EEG.icaact); % Empirical Mode Decomposition per component
%%
% Main Loop (Calculate power for 5 freq. bands of each component)
for i = 1:14
    
    %We need to extract a 6 x 17280 matrix from each component to pass to hht
    current_component_imfs = squeeze(allmodes(i,:,:));
    
    % Second step: Hilbert Spectrum Analysis
    % This is used to calculate the power of five freq. bands:
    % First, we'll calculate the hilbert spectrum of the IMFs
    [hs,f,t,imfinsf,imfinse] = hht(current_component_imfs,128);
    
    % Then the marginal spectrum can be calculated by integrating the hilbert
    % spectrum over time
    [A F] = mhs(imfinse,imfinsf,0,0.1,128);

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
    
    % Next, we'll find power of frequency bands we're interested in
    delta_sum_avg = mean(A(delta_idx));
    theta_sum_avg = mean(A(theta_idx));
    alpha_sum_avg = mean(A(alpha_idx));
    beta_sum_avg = mean(A(beta_idx));
    gamma_sum_avg = mean(A(gamma_idx));

    % Finally, we'll save these means in the grand matrix for each component and
    % loop to the next one
    allcomps(:,i) = horzcat(delta_sum_avg,theta_sum_avg,alpha_sum_avg,beta_sum_avg,gamma_sum_avg);
end
allshapesbaselinessubset(3,:,:) = allcomps; % add to baselines
%% Plot Features
out = rdivide(allshapesfeaturessubset,allshapesbaselinessubset); % normalize
%%
freq_strings = {'\delta','\theta','\alpha','\beta','\gamma'};
figure(1);
n = 1;
for m = 1:15 % loop through all subplots
        subplot(3,5,m);
        if m < 6
            title(freq_strings(m),'FontSize',16); % set and if statement for the first freq.
        end
        if mod(m,5) == 0
            h = 5;
        else
            h = mod(m,5);
        end
        topoplot(out(n,h,:),EEG.chanlocs(chans));
        if h == 5
           n = n + 1; 
        end
end
%% Feature Selection

% To evaluate which of the features
% provides the most useful information about the mental task, we used the Mann-Whitney-
% Wilcoxon (MWW) test [46]. We rank the features of the training data by MWW test in separate
% binary evaluations (each class vs. the other classes). We thus get a set of top features for a
% particular class. Top features for overall classification are selected by using a voting method
% among all sets of ranked features.

% This needs to be done class-wise, so we'll split up the dataset into each
% shape class

% p = ranksum(all_components,freq_axis); % Use Mann-Whitney-Wilcoxon test for finding best features

%% Classifier Training with LDA
%output = fitdiscr(p,);