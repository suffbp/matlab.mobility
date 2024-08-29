%% Sit, Stand, and turn features
% Assuming the following variables are defined:
% T_labeled.ML_denoised - The denoised ML acceleration data
% T_labeled.AP_denoised - The denoised AP acceleration data
% T_labeled.Vert_denoised - The denoised Vertical acceleration data
% newStand_start, newStand_end, newTurn_start, newTurn_end, newSit_start, newSit_end - Indices for segments
% newSamplingRate - The sampling rate of the data

% Define time vector
time_per_sample = 1 / newSamplingRate;

% Extract segments for stand, turn, and sit
stand_segment = T_labeled.ML_denoised(newStand_start:newStand_end);
turn_segment = T_labeled.ML_denoised(newTurn_start:newTurn_end);
sit_segment = T_labeled.ML_denoised(newSit_start:newSit_end);

% Calculate velocity for each segment using trapezoidal integration
stand_velocity = cumtrapz(time_per_sample * (1:length(stand_segment)), stand_segment);
turn_velocity = cumtrapz(time_per_sample * (1:length(turn_segment)), turn_segment);
sit_velocity = cumtrapz(time_per_sample * (1:length(sit_segment)), sit_segment);

% Calculate power as acceleration squared (this is a simplified version of power calculation)
stand_power = stand_segment.^2;
turn_power = turn_segment.^2;
sit_power = sit_segment.^2;

% Maximum Acceleration (MA)
stand_MA = max(abs(stand_segment));
turn_MA = max(abs(turn_segment));
sit_MA = max(abs(sit_segment));

% Maximum Velocity (MV)
stand_MV = max(abs(stand_velocity));
turn_MV = max(abs(turn_velocity));
sit_MV = max(abs(sit_velocity));

% Maximum Power (MP)
stand_MP = max(stand_power);
turn_MP = max(turn_power);
sit_MP = max(sit_power);

% Time to Complete (seconds)
stand_time_to_complete = length(stand_segment) * time_per_sample;
turn_time_to_complete = length(turn_segment) * time_per_sample;
sit_time_to_complete = length(sit_segment) * time_per_sample;

% Display Results
fprintf('Stand Period Features:\n');
fprintf('Maximum Acceleration: %.5f m/s^2\n', stand_MA);
fprintf('Maximum Velocity: %.5f m/s\n', stand_MV);
fprintf('Maximum Power: %.5f (a.u.)\n', stand_MP);
fprintf('Time to Complete: %.5f seconds\n\n', stand_time_to_complete);

fprintf('Turn Period Features:\n');
fprintf('Maximum Acceleration: %.5f m/s^2\n', turn_MA);
fprintf('Maximum Velocity: %.5f m/s\n', turn_MV);
fprintf('Maximum Power: %.5f (a.u.)\n', turn_MP);
fprintf('Time to Complete: %.5f seconds\n\n', turn_time_to_complete);

fprintf('Sit Period Features:\n');
fprintf('Maximum Acceleration: %.5f m/s^2\n', sit_MA);
fprintf('Maximum Velocity: %.5f m/s\n', sit_MV);
fprintf('Maximum Power: %.5f (a.u.)\n', sit_MP);
fprintf('Time to Complete: %.5f seconds\n', sit_time_to_complete);
