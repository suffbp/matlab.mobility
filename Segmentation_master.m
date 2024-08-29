%% Cell 1: Load & Plot Data
% cd '/Users/briansuffoletto/Desktop/Accel files'; % Change to the directory
% Specify the path to your Excel file
filePath = '/Users/bsuffoletto/Desktop/Accel/UPMC/e32.xls';
file_identifier = 'e32'; % Replace this with a dynamic identifier if processing multiple files

% Load the Excel file into a table
try
    T = readtable(filePath);
    disp('Data loaded successfully.');
    disp(T(1:5, :)); % Display the first five rows
catch ME
    disp('An error occurred while loading the file:');
    disp(ME.message);
end

% Assuming 'Seconds' is already numeric
timeSeconds = T.Time_s_; % Directly use the numeric 'Seconds' column

% Calculate the time differences in seconds
timeDiffs = diff(timeSeconds);

% Display the time differences to check
disp('Time differences (in seconds):');
disp(timeDiffs);

samplingRate = 1 / mean(timeDiffs); % Calculate the average sampling rate

% Display the calculated sampling rate
disp(['Calculated Sampling Rate: ', num2str(samplingRate), ' Hz']);

% Rename the columns
T = renamevars(T, {'X_m_s_2_', 'Y_m_s_2_', 'Z_m_s_2_'}, {'X', 'Y', 'Z'});

% Keep only the Time, X, Y, and Z columns
T = T(:, {'Time_s_', 'X', 'Y', 'Z'});

% Change back to the original MATLAB directory
cd '/Users/bsuffoletto/Desktop/Matlab/';

% Plotting the filtered data for X, Y, and Z
figure;
subplot(3, 1, 1);
plot(T.Time_s_, T.X);
title('Filtered Acceleration X over Time');
xlabel('Time (s)');
ylabel('Acceleration X (g)');

subplot(3, 1, 2);
plot(T.Time_s_, T.Y);
title('Filtered Acceleration Y over Time');
xlabel('Time (s)');
ylabel('Acceleration Y (g)');

subplot(3, 1, 3);
plot(T.Time_s_, T.Z);
title('Filtered Acceleration Z over Time');
xlabel('Time (s)');
ylabel('Acceleration Z (g)');

% Select the figure you want to interact with
subplot(3, 1, 1); % or 2 or 3 depending on which subplot you want to use

% Display a message to the user
disp('Select two points on the plot to calculate the time difference.');

% Get two points from the user
[x, ~] = ginput(2);

% Calculate the difference in time between the two selected points
timeDifference = abs(x(2) - x(1));

% Display the time difference
disp(['Time difference between the selected points: ', num2str(timeDifference), ' seconds']);

% Find the indices in the Time_s_ array corresponding to the selected time points
[~, idx1] = min(abs(T.Time_s_ - x(1)));
[~, idx2] = min(abs(T.Time_s_ - x(2)));

% Ensure idx1 is the smaller index and idx2 is the larger index
startIndex = min(idx1, idx2);
endIndex = max(idx1, idx2);

% Truncate the data series based on the selected indices
T_truncated = T(startIndex:endIndex, :);

% Save the truncated data series
save('truncated_data.mat', 'T_truncated');

% Display the truncated data for verification
figure;
subplot(3, 1, 1);
plot(T_truncated.Time_s_, T_truncated.X);
title('Truncated Acceleration X over Time');
xlabel('Time (s)');
ylabel('Acceleration X (g)');

subplot(3, 1, 2);
plot(T_truncated.Time_s_, T_truncated.Y);
title('Truncated Acceleration Y over Time');
xlabel('Time (s)');
ylabel('Acceleration Y (g)');

subplot(3, 1, 3);
plot(T_truncated.Time_s_, T_truncated.Z);
title('Truncated Acceleration Z over Time');
xlabel('Time (s)');
ylabel('Acceleration Z (g)');

% Clear 
clear idx1 idx2 startIndex T timeDifference x timeSeconds endIndex timeDiffs
%% Downsample to 50 Hz
newSamplingRate = 50;
[p, q] = rat(newSamplingRate / samplingRate); % Find rational approximation

% Check the data type and size of T.X, T.Y, and T.Z before resampling
if isnumeric(T_truncated.X) && isnumeric(T_truncated.Y) && isnumeric(T_truncated.Z)
    try
        X_resampled = resample(T_truncated.X, p, q);
        Y_resampled = resample(T_truncated.Y, p, q);
        Z_resampled = resample(T_truncated.Z, p, q);
    catch ME
        disp('Error during resampling:');
        disp(ME.message);
        return; % Exit the function if resampling fails
    end
else
    disp('Error: The data is not numeric.');
    return; % Exit the function if data is not numeric
end

% Create a new table for the resampled data
T_resampled = table();
T_resampled.X = X_resampled;
T_resampled.Y = Y_resampled;
T_resampled.Z = Z_resampled;

% Update the Time column based on the new sampling rate
numSamples = length(T_resampled.X); % Updated number of samples after downsampling
T_resampled.Time = (0:numSamples-1)' / newSamplingRate;

%drop extra vars
clear p q numSamples T 
clear X_resampled Y_resampled Z_resampled
clear timeDiffs timeSeconds


%% Center the X, Y, and Z variables by subtracting the mean
T_resampled.X_centered = T_resampled.X - mean(T_resampled.X);
T_resampled.Y_centered = T_resampled.Y - mean(T_resampled.Y);
T_resampled.Z_centered = T_resampled.Z - mean(T_resampled.Z);

%% Moving mean smooth
% Smoothing using moving mean with a specified window size
windowSize = 7; % Define the window size for moving mean, adjust as needed

T_resampled.X_smoothed = movmean(T_resampled.X_centered, windowSize);
T_resampled.Y_smoothed = movmean(T_resampled.Y_centered, windowSize);
T_resampled.Z_smoothed = movmean(T_resampled.Z_centered, windowSize);


%% Cell 3: Assign & relabel axes
[mlAxis, apAxis, vertAxis] = relabelAxes4(T_resampled);
T_labeled = createLabeledTable(apAxis, mlAxis, vertAxis);
clear apAxis mlAxis vertAxis
%% Correcting misalignment and de-noising data
% Extract data from T_labeled
api = T_labeled.AP;  % Antero-Posterior data
mli = T_labeled.ML;  % Medio-Lateral data
vi = T_labeled.Vert; % Vertical data

% Apply orientation correction to the accelerometer data
[ml_cor, v_cor, ap_cor] = acc_correction(mli, vi, api);

% Parameters for wavelet denoising
wname = 'dmey';      % Wavelet name (example: Discrete Meyer wavelet)
Nlevel = 10;         % Level of wavelet decomposition
threshapp = 's';     % Thresholding type ('s' for soft thresholding)
keepapp = 1;         % Flag to keep approximation coefficients

% Apply wavelet denoising to the corrected data
[ml] = denoiseacc(ml_cor, wname, Nlevel, threshapp, keepapp);
[v] = denoiseacc(v_cor, wname, Nlevel, threshapp, keepapp);
[ap] = denoiseacc(ap_cor, wname, Nlevel, threshapp, keepapp);

% Optional: Store the denoised data back into T_labeled
T_labeled.ML_denoised = ml;
T_labeled.Vert_denoised = v;
T_labeled.AP_denoised = ap;

% clear
clear ap ap_cor ap_denoised api ax hEnd hPeak hStart i keepapp ml ml_cor ml_denoised mli Nlevel v v_cor v_denoised vert_diff vi windowSize wname
clear threshapp

%% %% Sit-to-stand
% Assuming T_labeled.Vert contains the smoothed vertical axis data
Vert_smoothed = T_labeled.Vert_denoised; % Use the smoothed vertical axis data

% Determine the length of the data and calculate the first third
dataLength = length(Vert_smoothed);
oneThirdLength = floor(dataLength / 3);

% Identify the highest peak in the first third of the data
[~, Stand_peak] = max(Vert_smoothed(1:oneThirdLength));

% Calculate the difference (derivative) of the smoothed vertical data
vert_diff = diff(Vert_smoothed);

% Find Stand_start: Look for the negative-to-positive slope change before the peak
Stand_start = Stand_peak; % Initialize Stand_start
for i = Stand_peak-1:-1:2 % Start from Stand_peak and go backward, stopping before the first element
    if vert_diff(i-1) < 0 && vert_diff(i) > 0 % Slope change from negative to positive
        Stand_start = i;
        break; % Stop when the condition is met
    end
end

% Find Stand_end: Look for the negative-to-positive slope change after the peak
Stand_end = Stand_peak; % Initialize Stand_end
for i = Stand_peak+1:dataLength-1 % Start from Stand_peak and go forward, stopping before the last element
    if vert_diff(i) < 0 && vert_diff(i+1) > 0 % Slope change from negative to positive
        Stand_end = i + 1; % Adjust to match original data index
        break; % Stop when the condition is met
    end
end

% Plotting for visualization with manual adjustment
figure;
plot(Vert_smoothed, 'LineWidth', 1.5); % Plot the smoothed vertical data
hold on;
grid on;

% Add grid and improve y-axis resolution
ax = gca;
ax.YTick = min(Vert_smoothed):0.5:max(Vert_smoothed);

% Initial marking of start, peak, and end of stand-up motion
hStart = plot(Stand_start, Vert_smoothed(Stand_start), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Start of rise
hPeak = plot(Stand_peak, Vert_smoothed(Stand_peak), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');    % Peak of rise
hEnd = plot(Stand_end, Vert_smoothed(Stand_end), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');       % End of rise

% Labeling the plot
title('Detection of Stand-Up Motion with Slope Changes');
xlabel('Sample Index');
ylabel('Acceleration (m/s^2)');
legend('Smoothed Vertical Data', 'Start of Rise', 'Peak of Rise', 'End of Rise');
hold off;

% Prompt the user to manually adjust the points
disp('Click to select new Stand_start point on the plot...');
[newStand_start, ~] = ginput(1); % User clicks to select new Stand_start point
Stand_start = round(newStand_start); % Update Stand_start with the new selected index

disp('Click to select new Stand_end point on the plot...');
[newStand_end, ~] = ginput(1); % User clicks to select new Stand_end point
Stand_end = round(newStand_end); % Update Stand_end with the new selected index

% Update the plot with manually adjusted points
set(hStart, 'XData', Stand_start, 'YData', Vert_smoothed(Stand_start)); % Update Stand_start marker
set(hEnd, 'XData', Stand_end, 'YData', Vert_smoothed(Stand_end)); % Update Stand_end marker

% Refresh the plot
drawnow;

disp(['New Stand_start: ', num2str(Stand_start)]);
disp(['New Stand_end: ', num2str(Stand_end)]);

% clear
clear Stand_start Stand_end

%% Stand to Sit
% Determine the length of the data and calculate the last third
lastThirdStart = 2 * oneThirdLength + 1; % Start index for the last third of the data

% Focus on the last third of the smoothed data
Vert_lastThird = Vert_smoothed(lastThirdStart:end);

% Identify the highest peak in the last third of the data (Sit_peak)
[~, peak_index_relative] = max(Vert_lastThird);
Sit_peak = peak_index_relative + lastThirdStart - 1; % Adjust to match original data index

% Calculate the difference (derivative) of the smoothed vertical data
vert_diff = diff(Vert_smoothed);

% Find Sit_start: Look for the positive-to-negative slope change before the peak
Sit_start = Sit_peak; % Initialize Sit_start
for i = Sit_peak-1:-1:lastThirdStart % Start from Sit_peak and go backward
    if vert_diff(i-1) > 0 && vert_diff(i) < 0 % Slope change from positive to negative
        Sit_start = i;
        break; % Stop when the condition is met
    end
end

% Find Sit_end: Look for the positive-to-negative slope change after the peak
Sit_end = Sit_peak; % Initialize Sit_end
for i = Sit_peak+1:dataLength-2 % Adjust loop to stop before out-of-bounds
    if vert_diff(i) > 0 && vert_diff(i+1) < 0 % Slope change from positive to negative
        Sit_end = i + 1; % Adjust to match original data index
        break; % Stop when the condition is met
    end
end

%Plotting for visualization with manual adjustment
figure;
plot(Vert_smoothed, 'LineWidth', 1.5); % Plot the smoothed vertical data
hold on;
grid on;

% Add grid and improve y-axis resolution
ax = gca;
ax.YTick = min(Vert_smoothed):0.5:max(Vert_smoothed);

% Initial marking of start, peak, and end of sit-to-stand motion
hStart = plot(Sit_start, Vert_smoothed(Sit_start), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Sit start
hPeak = plot(Sit_peak, Vert_smoothed(Sit_peak), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');    % Sit peak
hEnd = plot(Sit_end, Vert_smoothed(Sit_end), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');       % Sit end

% Labeling the plot
title('Detection of Sit-to-Stand Motion with Slope Changes');
xlabel('Sample Index');
ylabel('Acceleration (m/s^2)');
legend('Smoothed Vertical Data', 'Sit Start', 'Sit Peak', 'Sit End');
hold off;

% Prompt the user to manually adjust the points
disp('Click to select new Sit_start point on the plot...');
[newSit_start, ~] = ginput(1); % User clicks to select new Sit_start point
Sit_start = round(newSit_start); % Update Sit_start with the new selected index

disp('Click to select new Sit_end point on the plot...');
[newSit_end, ~] = ginput(1); % User clicks to select new Sit_end point
Sit_end = round(newSit_end); % Update Sit_end with the new selected index

% Update the plot with manually adjusted points
set(hStart, 'XData', Sit_start, 'YData', Vert_smoothed(Sit_start)); % Update Sit_start marker
set(hEnd, 'XData', Sit_end, 'YData', Vert_smoothed(Sit_end)); % Update Sit_end marker

% Refresh the plot
drawnow;

disp(['New Sit_start: ', num2str(Sit_start)]);
disp(['New Sit_end: ', num2str(Sit_end)]);

% Make integers & clean
% Ensure indices are integers and within bounds
newStand_start = round(newStand_start);    
newStand_end = round(newStand_end); 
newSit_start = round(newSit_start); 
newSit_end = round(newSit_end);

clear Sit_start Sit_End i hPeak hEnd ax hStart lastThirdStart peak_index_relative vert_diff Vert_lastThird

%% Turn

% Define the segment of interest in T_labeled (assuming already specified)
segment_indices = newStand_end:newSit_start;  % Example indices, adjust as needed

% Extract the ML (mediolateral) denoised data (acceleration)
ML_acceleration = T_labeled.ML_denoised(segment_indices);  % Acceleration (ML axis)
time = (0:length(ML_acceleration)-1) / newSamplingRate;   % Time vector

% Calculate velocity using trapezoidal integration of acceleration
ML_velocity = cumtrapz(time, ML_acceleration);

% Define sliding window parameters
window_size = round(newSamplingRate * 1); % 1-second window, adjust as needed
window_step = 1; % Step size for the sliding window

% Initialize variables to store deflection values
deflections = zeros(1, length(ML_velocity) - window_size + 1);

% Compute the sum of absolute changes in velocity within each window
for i = 1:window_step:length(deflections)
    window_end = i + window_size - 1;
    if window_end > length(ML_velocity)
        break;
    end
    window_data = ML_velocity(i:window_end);
    deflections(i) = sum(abs(diff(window_data)));
end

% Define time vector for deflections (aligned with the middle of each window)
time_deflections = time(1:length(deflections));

% Find the window with the largest deflection
[~, max_deflection_index] = max(deflections);
turn_start = max_deflection_index;  % Largest deflection starts here

% Define a fitting window around the largest deflection for curve fitting
fit_start = max(turn_start - 5, 1);  % Slightly before the detected start
fit_end = min(turn_start + window_size - 1, length(ML_velocity));  % Include whole deflection

% Extract deflection data for fitting
fit_time = time(fit_start:fit_end);
fit_velocity = ML_velocity(fit_start:fit_end);

% Fit a polynomial curve to the deflection data
poly_order = 2; % Quadratic fit
p = polyfit(fit_time, fit_velocity, poly_order);

% Generate fitted curve data
fit_curve = polyval(p, fit_time);

% Identify turn_start and turn_end based on fitted curve
% Use first derivative to find start of upward curve and peak
fit_curve_deriv = diff(fit_curve);
[~, turn_start_rel] = max(fit_curve_deriv);  % Start of upward curve
[~, turn_end_rel] = max(fit_curve);  % Peak of the curve

turn_start = fit_start + turn_start_rel - 1;
turn_end = fit_start + turn_end_rel - 1;

% Plotting Acceleration, Velocity Detection
figure;

% Plot ML Acceleration
subplot(2, 1, 1);
plot(time, ML_acceleration, 'LineWidth', 1.5);
title('ML Acceleration');
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
grid on;

% Plot ML Velocity
subplot(2, 1, 2);
plot(time, ML_velocity, 'LineWidth', 1.5);
hold on;
hTurnStart = plot(time(turn_start), ML_velocity(turn_start), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Turn Start');
hTurnEnd = plot(time(turn_end), ML_velocity(turn_end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'Turn End');
title('ML Velocity');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('show');
grid on;

% Prompt the user to manually adjust the points
disp('Click to select new Turn Start point on the plot...');
[newTurn_start, ~] = ginput(1); % User clicks to select new Turn Start point
turn_start = round(newTurn_start * newSamplingRate); % Update turn_start with the new selected index

disp('Click to select new Turn End point on the plot...');
[newTurn_end, ~] = ginput(1); % User clicks to select new Turn End point
turn_end = round(newTurn_end * newSamplingRate); % Update turn_end with the new selected index

% Update the plot with manually adjusted points
set(hTurnStart, 'XData', time(turn_start), 'YData', ML_velocity(turn_start)); % Update Turn Start marker
set(hTurnEnd, 'XData', time(turn_end), 'YData', ML_velocity(turn_end)); % Update Turn End marker

% Refresh the plot
drawnow;

% Display the results
disp(['New Turn Start: ', num2str(turn_start)]);
disp(['New Turn End: ', num2str(turn_end)]);
fprintf('Turn Start Time: %.2f seconds\n', time(turn_start));
fprintf('Turn End Time: %.2f seconds\n', time(turn_end));

% Adjust the turn_start and turn_end to reference the full T_labeled dataset
turn_start_global = turn_start + newStand_end - 1;
turn_end_global = turn_end + newStand_end - 1;

% Clean
clear turn_start turn_end turn_start_rel turn_end_rel fit_curve 
clear fit_curve_deriv fit_end fit_start fit_time fit_velocity 
clear hTurnEnd hTurnStart i max_deflection_index p poly_order 
clear segment_indices time_deflections Sit_peak Stand_peak Sit_end 
clear window_step window_size window_end window_data ML_velocity deflections ML_acceleration 
clear newTurn_end newTurn_start
%% Plot check
% Plotting the key events on ML, AP, and Vert denoised data

% Extract the data
ML_denoised = T_labeled.ML_denoised;  % ML (Medio-Lateral) axis
AP_denoised = T_labeled.AP_denoised;  % AP (Antero-Posterior) axis
Vert_denoised = T_labeled.Vert_denoised; % Vert (Vertical) axis
time = (0:length(ML_denoised)-1) / newSamplingRate; % Time vector

% Plot ML Axis Data
figure;
subplot(3, 1, 1);
plot(time, ML_denoised, 'LineWidth', 1.5);
hold on;
plot(time(newStand_start), ML_denoised(newStand_start), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Stand Start');
plot(time(newStand_end), ML_denoised(newStand_end), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Stand End');
plot(time(turn_start_global), ML_denoised(turn_start_global), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', 'Turn Start');
plot(time(turn_end_global), ML_denoised(turn_end_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Turn End');
plot(time(newSit_start), ML_denoised(newSit_start), 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c', 'DisplayName', 'Sit Start');
plot(time(newSit_end), ML_denoised(newSit_end), 'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'DisplayName', 'Sit End');
title('ML Denoised Data with Key Events');
xlabel('Time (s)');
ylabel('Acceleration (ML)');
legend('show');
grid on;
hold off;

% Plot AP Axis Data
subplot(3, 1, 2);
plot(time, AP_denoised, 'LineWidth', 1.5);
hold on;
plot(time(newStand_start), AP_denoised(newStand_start), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Stand Start');
plot(time(newStand_end), AP_denoised(newStand_end), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Stand End');
plot(time(turn_start_global), AP_denoised(turn_start_global), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', 'Turn Start');
plot(time(turn_end_global), AP_denoised(turn_end_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Turn End');
plot(time(newSit_start), AP_denoised(newSit_start), 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c', 'DisplayName', 'Sit Start');
plot(time(newSit_end), AP_denoised(newSit_end), 'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'DisplayName', 'Sit End');
title('AP Denoised Data with Key Events');
xlabel('Time (s)');
ylabel('Acceleration (AP)');
legend('show');
grid on;
hold off;

% Plot Vert Axis Data
subplot(3, 1, 3);
plot(time, Vert_denoised, 'LineWidth', 1.5);
hold on;
plot(time(newStand_start), Vert_denoised(newStand_start), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'DisplayName', 'Stand Start');
plot(time(newStand_end), Vert_denoised(newStand_end), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'DisplayName', 'Stand End');
plot(time(turn_start_global), Vert_denoised(turn_start_global), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm', 'DisplayName', 'Turn Start');
plot(time(turn_end_global), Vert_denoised(turn_end_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Turn End');
plot(time(newSit_start), Vert_denoised(newSit_start), 'co', 'MarkerSize', 8, 'MarkerFaceColor', 'c', 'DisplayName', 'Sit Start');
plot(time(newSit_end), Vert_denoised(newSit_end), 'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y', 'DisplayName', 'Sit End');
title('Vert Denoised Data with Key Events');
xlabel('Time (s)');
ylabel('Acceleration (Vert)');
legend('show');
grid on;
hold off;

%% %% Gait segmentation 
newSamplingRate = 50;
[pslt, pslh, psrt, psrh] = strdesegm(T_labeled, newStand_end, newSit_start, turn_start_global, turn_end_global, newSamplingRate);

%% Adjust gait event indices to match T_labeled

% Assume pslt, pslh, psrt, psrh are obtained from stridesegmentation_from_segment function

% Adjust indices to match the original data indices in T_labeled
pslt_global = pslt + newStand_end - 1; % Adjusting to global indices
pslh_global = pslh + newStand_end - 1; % Adjusting to global indices
psrt_global = psrt + newStand_end - 1; % Adjusting to global indices
psrh_global = psrh + newStand_end - 1; % Adjusting to global indices

% Define the overall valid segment (excluding the turn)
valid_indices = (pslt_global >= newStand_end & pslt_global <= newSit_start) & ...
                ~(pslt_global >= turn_start_global & pslt_global <= turn_end_global);
                
pslt_global = pslt_global(valid_indices);

valid_indices = (pslh_global >= newStand_end & pslh_global <= newSit_start) & ...
                ~(pslh_global >= turn_start_global & pslh_global <= turn_end_global);
                
pslh_global = pslh_global(valid_indices);

valid_indices = (psrt_global >= newStand_end & psrt_global <= newSit_start) & ...
                ~(psrt_global >= turn_start_global & psrt_global <= turn_end_global);
                
psrt_global = psrt_global(valid_indices);

valid_indices = (psrh_global >= newStand_end & psrh_global <= newSit_start) & ...
                ~(psrh_global >= turn_start_global & psrh_global <= turn_end_global);
                
psrh_global = psrh_global(valid_indices);

%% PLot

% Check if indices are within the valid range
dataLength = height(T_labeled); % Assuming T_labeled is a table
if newStand_end < 1 || newStand_end > dataLength || newSit_start < 1 || newSit_start > dataLength
    error('Indices are out of bounds. Check newStand_end and newSit_start.');
end

% Ensure that newStand_end is less than newSit_start
if newStand_end >= newSit_start
    error('newStand_end must be less than newSit_start.');
end

% Plot the AP, ML, and Vert data segments with gait events
figure;

% AP Segment Plot
subplot(3, 1, 1);
plot(newStand_end:newSit_start, T_labeled.AP_denoised(newStand_end:newSit_start), 'LineWidth', 1.5);
hold on;
plot(pslt_global, T_labeled.AP_denoised(pslt_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Toe-Off'); % Open circle for toe-off
plot(pslh_global, T_labeled.AP_denoised(pslh_global), 'rx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Heel-Strike'); % 'x' for heel-strike
plot(psrt_global, T_labeled.AP_denoised(psrt_global), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Toe-Off'); % Open circle for toe-off
plot(psrh_global, T_labeled.AP_denoised(psrh_global), 'bx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Heel-Strike'); % 'x' for heel-strike
title('AP Segment with Gait Events');
xlabel('Sample Index');
ylabel('Acceleration (AP)');
legend('show');
hold off;

% ML Segment Plot
subplot(3, 1, 2);
plot(newStand_end:newSit_start, T_labeled.ML_denoised(newStand_end:newSit_start), 'LineWidth', 1.5);
hold on;
plot(pslt_global, T_labeled.ML_denoised(pslt_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Toe-Off'); % Open circle for toe-off
plot(pslh_global, T_labeled.ML_denoised(pslh_global), 'rx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Heel-Strike'); % 'x' for heel-strike
plot(psrt_global, T_labeled.ML_denoised(psrt_global), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Toe-Off'); % Open circle for toe-off
plot(psrh_global, T_labeled.ML_denoised(psrh_global), 'bx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Heel-Strike'); % 'x' for heel-strike
title('ML Segment with Gait Events');
xlabel('Sample Index');
ylabel('Acceleration (ML)');
legend('show');
hold off;

% Vert Segment Plot
subplot(3, 1, 3);
plot(newStand_end:newSit_start, T_labeled.Vert_denoised(newStand_end:newSit_start), 'LineWidth', 1.5);
hold on;
plot(pslt_global, T_labeled.Vert_denoised(pslt_global), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Toe-Off'); % Open circle for toe-off
plot(pslh_global, T_labeled.Vert_denoised(pslh_global), 'rx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Left Heel-Strike'); % 'x' for heel-strike
plot(psrt_global, T_labeled.Vert_denoised(psrt_global), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Toe-Off'); % Open circle for toe-off
plot(psrh_global, T_labeled.Vert_denoised(psrh_global), 'bx', 'MarkerSize', 10, 'MarkerFaceColor', 'none', 'DisplayName', 'Right Heel-Strike'); % 'x' for heel-strike
title('Vert Segment with Gait Events');
xlabel('Sample Index');
ylabel('Acceleration (Vert)');
legend('show');
hold off;

