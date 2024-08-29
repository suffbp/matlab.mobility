% Assuming pslt, pslh, psrt, and psrh are vectors with the toe-off and heel-strike times (in samples)

% Convert sample indices to time using sampling frequency
Fs = newSamplingRate;  % Sampling frequency (Hz)
pslt_time = pslt_global / Fs;
pslh_time = pslh_global / Fs;
psrt_time = psrt_global / Fs;
psrh_time = psrh_global / Fs;

%% Calculate Step Time (heel-strike to heel-strike of opposite foot)
% Combine and sort heel-strike times with labels indicating left (L) or right (R)
all_heel_strikes = [pslh_global, psrh_global];
heel_labels = [repmat('L', 1, length(pslh_global)), repmat('R', 1, length(psrh_global))];

% Sort heel strikes and labels together
[all_heel_strikes, sort_indices] = sort(all_heel_strikes);
heel_labels = heel_labels(sort_indices);

% Initialize step_time array
step_time = [];

% Calculate Step Time (heel-strike to heel-strike of opposite foot)
for i = 1:length(all_heel_strikes) - 1
    if heel_labels(i) ~= heel_labels(i + 1)
        % If the labels are different, calculate the step time
        step_time = [step_time, all_heel_strikes(i + 1) - all_heel_strikes(i)];
    end
    % If the labels are the same, skip this pair
end

% Display the calculated step times
disp('Step times (heel-strike to heel-strike of opposite foot):');
disp(step_time);

% Calculate Mean Times 
mean_step_time = mean(step_time);

% Define the sampling rate
sampling_rate = 50;  % Hz
time_per_sample = 1 / sampling_rate;  % Time in seconds per sample

% Assuming step_time is calculated as differences in row numbers
% Calculate the mean of step_time
mean_step_time = mean(step_time);

% Convert mean_step_time to actual time (seconds)
mean_step_time_in_seconds = mean_step_time * time_per_sample;

% Display the converted mean step time
disp('Mean step time in seconds (heel-strike to heel-strike of opposite foot):');
disp(mean_step_time_in_seconds);

%% Calculate Stride Time (heel-strike to heel-strike of the same foot)
% Initialize stride time arrays
valid_left_stride_time = [];
valid_right_stride_time = [];

% Check for valid left stride times
for i = 2:length(pslh_global)
    % Find heel strikes between the current and the previous left heel strike
    intervening_right = psrh_global(psrh_global > pslh_global(i-1) & psrh_global < pslh_global(i));
    
    if ~isempty(intervening_right)
        % If there's at least one right heel strike between two left heel strikes, it's a valid stride
        valid_left_stride_time = [valid_left_stride_time, pslh_global(i) - pslh_global(i-1)];
    end
end

% Check for valid right stride times
for i = 2:length(psrh_global)
    % Find heel strikes between the current and the previous right heel strike
    intervening_left = pslh_global(pslh_global > psrh_global(i-1) & pslh_global < psrh_global(i));
    
    if ~isempty(intervening_left)
        % If there's at least one left heel strike between two right heel strikes, it's a valid stride
        valid_right_stride_time = [valid_right_stride_time, psrh_global(i) - psrh_global(i-1)];
    end
end

% Display the valid stride times
disp('Valid Left Stride Times:');
disp(valid_left_stride_time);

disp('Valid Right Stride Times:');
disp(valid_right_stride_time);

mean_left_stride_time = mean(valid_left_stride_time);
mean_right_stride_time = mean(valid_right_stride_time);
mean_combined_stride_time = (mean_left_stride_time + mean_right_stride_time) / 2;

% Convert mean stride times to actual time (seconds)
mean_left_stride_time_in_seconds = mean_left_stride_time * time_per_sample;
mean_right_stride_time_in_seconds = mean_right_stride_time * time_per_sample;
mean_combined_stride_time_in_seconds = mean_combined_stride_time * time_per_sample;

% Display the converted mean stride times
disp('Mean left stride time in seconds:');
disp(mean_left_stride_time_in_seconds);

disp('Mean right stride time in seconds:');
disp(mean_right_stride_time_in_seconds);

disp('Mean combined stride time in seconds:');
disp(mean_combined_stride_time_in_seconds);

%% Calculate Stance Time (heel-strike to toe-off of the same foot)
% Define the maximum allowable distance (in samples) between heel strike and toe-off
max_distance = 150;

% Separate left foot heel strikes and toe-offs
left_heel_strikes = pslh_global;
left_toe_offs = pslt_global;

% Separate right foot heel strikes and toe-offs
right_heel_strikes = psrh_global;
right_toe_offs = psrt_global;

% Initialize arrays to store valid stance times
valid_left_stance_time = [];
valid_right_stance_time = [];

% Calculate valid left stance times (heel strike to toe-off on the same foot)
for i = 1:length(left_heel_strikes)
    % Find the next toe-off after the current left heel strike
    next_left_toe_off_idx = find(left_toe_offs > left_heel_strikes(i), 1);
    
    if ~isempty(next_left_toe_off_idx)
        % Check if the next toe-off is within the max_distance
        if (left_toe_offs(next_left_toe_off_idx) - left_heel_strikes(i)) <= max_distance
            % Calculate and store the valid left stance time
            stance_time = left_toe_offs(next_left_toe_off_idx) - left_heel_strikes(i);
            valid_left_stance_time = [valid_left_stance_time, stance_time];
        end
    end
end

% Calculate valid right stance times (heel strike to toe-off on the same foot)
for i = 1:length(right_heel_strikes)
    % Find the next toe-off after the current right heel strike
    next_right_toe_off_idx = find(right_toe_offs > right_heel_strikes(i), 1);
    
    if ~isempty(next_right_toe_off_idx)
        % Check if the next toe-off is within the max_distance
        if (right_toe_offs(next_right_toe_off_idx) - right_heel_strikes(i)) <= max_distance
            % Calculate and store the valid right stance time
            stance_time = right_toe_offs(next_right_toe_off_idx) - right_heel_strikes(i);
            valid_right_stance_time = [valid_right_stance_time, stance_time];
        end
    end
end

% Calculate mean stance times
mean_left_stance_time = mean(valid_left_stance_time);
mean_right_stance_time = mean(valid_right_stance_time);
mean_combined_stance_time = (mean_left_stance_time + mean_right_stance_time) / 2;

% Define the sampling rate
sampling_rate = 50;  % Hz
time_per_sample = 1 / sampling_rate;  % Time in seconds per sample

% Convert mean stance times to actual time (seconds)
mean_left_stance_time_in_seconds = mean_left_stance_time * time_per_sample;
mean_right_stance_time_in_seconds = mean_right_stance_time * time_per_sample;
mean_combined_stance_time_in_seconds = mean_combined_stance_time * time_per_sample;

% Display the converted mean stance times
disp('Mean left stance time in seconds:');
disp(mean_left_stance_time_in_seconds);

disp('Mean right stance time in seconds:');
disp(mean_right_stance_time_in_seconds);

disp('Mean combined stance time in seconds:');
disp(mean_combined_stance_time_in_seconds);


%% Calculate Swing Time (toe-off to heel-strike of the same foot)
% Define the maximum allowable distance (in samples) between toe-off and heel strike
max_distance_swing = 100; % You may adjust this as per your dataset characteristics

% Initialize arrays to store valid swing times
valid_left_swing_time = [];
valid_right_swing_time = [];

% Calculate valid left swing times (toe-off to next heel strike on the same foot)
for i = 1:length(left_toe_offs)
    % Find the next heel strike after the current left toe-off
    next_left_heel_strike_idx = find(left_heel_strikes > left_toe_offs(i), 1);
    
    if ~isempty(next_left_heel_strike_idx)
        % Check if the next heel strike is within the max_distance_swing
        if (left_heel_strikes(next_left_heel_strike_idx) - left_toe_offs(i)) <= max_distance_swing
            % Calculate and store the valid left swing time
            swing_time = left_heel_strikes(next_left_heel_strike_idx) - left_toe_offs(i);
            valid_left_swing_time = [valid_left_swing_time, swing_time];
        end
    end
end

% Calculate valid right swing times (toe-off to next heel strike on the same foot)
for i = 1:length(right_toe_offs)
    % Find the next heel strike after the current right toe-off
    next_right_heel_strike_idx = find(right_heel_strikes > right_toe_offs(i), 1);
    
    if ~isempty(next_right_heel_strike_idx)
        % Check if the next heel strike is within the max_distance_swing
        if (right_heel_strikes(next_right_heel_strike_idx) - right_toe_offs(i)) <= max_distance_swing
            % Calculate and store the valid right swing time
            swing_time = right_heel_strikes(next_right_heel_strike_idx) - right_toe_offs(i);
            valid_right_swing_time = [valid_right_swing_time, swing_time];
        end
    end
end

% Calculate mean swing times
mean_left_swing_time = mean(valid_left_swing_time);
mean_right_swing_time = mean(valid_right_swing_time);
mean_combined_swing_time = (mean_left_swing_time + mean_right_swing_time) / 2;

% Convert mean swing times to actual time (seconds)
mean_left_swing_time_in_seconds = mean_left_swing_time * time_per_sample;
mean_right_swing_time_in_seconds = mean_right_swing_time * time_per_sample;
mean_combined_swing_time_in_seconds = mean_combined_swing_time * time_per_sample;

% Display the converted mean swing times
disp('Mean left swing time in seconds:');
disp(mean_left_swing_time_in_seconds);

disp('Mean right swing time in seconds:');
disp(mean_right_swing_time_in_seconds);

disp('Mean combined swing time in seconds:');
disp(mean_combined_swing_time_in_seconds);

%% Calculate DST
% Initialize double support time array
double_support_time = [];

% Loop through all left heel strikes and find corresponding double support periods
for i = 1:length(left_heel_strikes)
    % Check for double support: Right foot is still in stance after left heel strike
    next_right_toe_off_idx = find(right_toe_offs > left_heel_strikes(i), 1);
    if ~isempty(next_right_toe_off_idx) && right_toe_offs(next_right_toe_off_idx) > left_heel_strikes(i)
        % Calculate time until right toe-off
        double_support = right_toe_offs(next_right_toe_off_idx) - left_heel_strikes(i);
        double_support_time = [double_support_time, double_support];
    end
end

% Loop through all right heel strikes and find corresponding double support periods
for i = 1:length(right_heel_strikes)
    % Check for double support: Left foot is still in stance after right heel strike
    next_left_toe_off_idx = find(left_toe_offs > right_heel_strikes(i), 1);
    if ~isempty(next_left_toe_off_idx) && left_toe_offs(next_left_toe_off_idx) > right_heel_strikes(i)
        % Calculate time until left toe-off
        double_support = left_toe_offs(next_left_toe_off_idx) - right_heel_strikes(i);
        double_support_time = [double_support_time, double_support];
    end
end

% Calculate total double support time in the gait cycle
total_double_support_time = sum(double_support_time);

% Convert total double support time to seconds
total_double_support_time_in_seconds = total_double_support_time * time_per_sample;

% Calculate mean double support time per cycle (if needed)
mean_double_support_time = mean(double_support_time);
mean_double_support_time_in_seconds = mean_double_support_time * time_per_sample;

% Display the mean double support time in seconds
disp('Mean double support time per cycle in seconds:');
disp(mean_double_support_time_in_seconds);

% Calculate the standard deviation of double support time
std_double_support_time = std(double_support_time);
std_double_support_time_in_seconds = std_double_support_time * time_per_sample;

% Display the standard deviation of double support time in seconds
disp('Standard deviation of double support time per cycle in seconds:');
disp(std_double_support_time_in_seconds);

%% Convert Stride, Swing, and Stance Times to Seconds
valid_left_stride_time_in_seconds = valid_left_stride_time * time_per_sample;
valid_right_stride_time_in_seconds = valid_right_stride_time * time_per_sample;

valid_left_swing_time_in_seconds = valid_left_swing_time * time_per_sample;
valid_right_swing_time_in_seconds = valid_right_swing_time * time_per_sample;

valid_left_stance_time_in_seconds = valid_left_stance_time * time_per_sample;
valid_right_stance_time_in_seconds = valid_right_stance_time * time_per_sample;

%% Calculate Asymmetry
stride_time_asymmetry = abs(mean(valid_left_stride_time_in_seconds) - mean(valid_right_stride_time_in_seconds));
swing_time_asymmetry = abs(mean(valid_left_swing_time_in_seconds) - mean(valid_right_swing_time_in_seconds));
stance_time_asymmetry = abs(mean(valid_left_stance_time_in_seconds) - mean(valid_right_stance_time_in_seconds));

%% Calculate Variability (Standard Deviation)
left_stride_time_std = std(valid_left_stride_time_in_seconds);
right_stride_time_std = std(valid_right_stride_time_in_seconds);
combined_stride_time_std = sqrt((left_stride_time_std^2 + right_stride_time_std^2) / 2);

left_swing_time_std = std(valid_left_swing_time_in_seconds);
right_swing_time_std = std(valid_right_swing_time_in_seconds);
combined_swing_time_std = sqrt((left_swing_time_std^2 + right_swing_time_std^2) / 2);

left_stance_time_std = std(valid_left_stance_time_in_seconds);
right_stance_time_std = std(valid_right_stance_time_in_seconds);
combined_stance_time_std = sqrt((left_stance_time_std^2 + right_stance_time_std^2) / 2);

%% Display the Results
disp('Stride Time Asymmetry (seconds):');
disp(stride_time_asymmetry);

disp('Swing Time Asymmetry (seconds):');
disp(swing_time_asymmetry);

disp('Stance Time Asymmetry (seconds):');
disp(stance_time_asymmetry);

disp('Combined Stride Time Standard Deviation (seconds):');
disp(combined_stride_time_std);

disp('Combined Swing Time Standard Deviation (seconds):');
disp(combined_swing_time_std);

disp('Combined Stance Time Standard Deviation (seconds):');
disp(combined_stance_time_std);

