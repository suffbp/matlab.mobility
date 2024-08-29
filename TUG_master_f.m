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

% Calculate Lyapunov Exponent
LyE = calculateLyapunovExponent(walking_segment, newSamplingRate);

% Display the result
fprintf('Lyapunov Exponent for walking segment: %.4f\n', LyE);

% Function to calculate Lyapunov Exponent using the Wolf algorithm
function LyE = calculateLyapunovExponent(data, Fs)
    % Parameters for the LyE calculation
    m = 3;                % Embedding dimension
    tau = 1;              % Time delay
    epsilon = 0.1;        % Threshold for nearest neighbors
    T = 0.1 * Fs;         % Maximum separation between points
    
    % Phase space reconstruction
    X = phaseSpaceReconstruction(data, m, tau);
    N = size(X, 1);
    
    % Calculate Lyapunov Exponent
    d_sum = 0;    % Sum of divergence distances
    count = 0;    % Count of valid points
    
    for i = 1:N-T
        % Find nearest neighbor
        d_min = Inf;
        j_min = 0;
        for j = i+1:N-T
            d_ij = norm(X(i, :) - X(j, :));
            if d_ij < d_min && d_ij > epsilon
                d_min = d_ij;
                j_min = j;
            end
        end
        
        % Calculate divergence if a valid neighbor is found
        if j_min > 0
            d_t = norm(X(i+T, :) - X(j_min+T, :));
            d_sum = d_sum + log(d_t / d_min);
            count = count + 1;
        end
    end
    
    % Calculate Lyapunov Exponent
    LyE = (d_sum / count) / T;
end

% Function for phase space reconstruction
function X = phaseSpaceReconstruction(data, m, tau)
    N = length(data) - (m - 1) * tau;
    X = zeros(N, m);
    for i = 1:m
        X(:, i) = data((1:N) + (i-1)*tau);
    end
end

%%From: The code follows the method outlined in "Multiscale and Shannon entropies during gait as fall risk predictors—A prospective study."
%%ALso: Amirpourabasi A, Lamb SE, Chow JY, Williams GKR. Nonlinear Dynamic Measures of Walking in Healthy Older Adults: A Systematic Scoping Review. Sensors (Basel). 2022 Jun 10;22(12):4408. doi: 10.3390/s22124408.
% Parameters for the calculation
newSamplingRate = 50; % Adjust based on your data
segment_indices = round(newStand_end):round(newSit_start); % Indices for walking segment

% Extract walking segment from T_labeled
walking_segment = T_labeled.ML_denoised(segment_indices); % Use Mediolateral axis for example

% Define MSE parameters
max_scale = 20; % Maximum scale for MSE, adjust as needed
m = 2; % Embedding dimension for sample entropy
r = 0.15 * std(walking_segment); % Tolerance (typically 0.1-0.25 * std of data)

% Calculate Multiscale Entropy
mse_values = calculateMultiscaleEntropy(walking_segment, m, r, max_scale);

% Replace infinite MSE values with NaN
mse_values_cleaned = mse_values;
mse_values_cleaned(isinf(mse_values_cleaned)) = NaN;


% Calculate CI using trapezoidal integration on valid data
valid_mse_values = mse_values_cleaned(~isnan(mse_values_cleaned));
CI = trapz(1:length(valid_mse_values), valid_mse_values);

% Display the CI value
disp('Index of Complexity (CI) using cleaned MSE values:');
disp(CI);

% Function to calculate Multiscale Entropy
function mse_values = calculateMultiscaleEntropy(data, m, r, max_scale)
    mse_values = zeros(1, max_scale);
    for scale = 1:max_scale
        % Coarse-grain the data at current scale
        coarse_grained_data = coarseGrain(data, scale);
        
        % Calculate sample entropy for the coarse-grained data
        mse_values(scale) = sampleEntropy(coarse_grained_data, m, r);
    end
end

% Function for coarse-graining the data
function coarse_data = coarseGrain(data, scale)
    len = floor(length(data) / scale);
    coarse_data = zeros(1, len);
    for i = 1:len
        coarse_data(i) = mean(data((i-1)*scale+1:i*scale));
    end
end

% Function to calculate Sample Entropy (SampEn)
function se = sampleEntropy(data, m, r)
    N = length(data);
    A = 0; B = 0;
    for i = 1:N-m
        template = data(i:i+m-1);
        count_A = 0;
        count_B = 0;
        for j = 1:N-m
            if i ~= j
                compare = data(j:j+m-1);
                dist = max(abs(template - compare));
                if dist < r
                    count_B = count_B + 1;
                    if abs(data(i+m) - data(j+m)) < r
                        count_A = count_A + 1;
                    end
                end
            end
        end
        A = A + count_A;
        B = B + count_B;
    end
    % Avoid log of zero by setting a small lower limit for B
    if B == 0
        B = 1;
    end
    se = -log(A / B);
end

% Define the segment of interest (newStand_end to newSit_start)
segment_indices = newStand_end:newSit_start;

% Extract the ML, AP, and Vert denoised data for the gait segment
ML_segment = T_labeled.ML_denoised(segment_indices);
AP_segment = T_labeled.AP_denoised(segment_indices);
Vert_segment = T_labeled.Vert_denoised(segment_indices);

% Sampling rate (Hz)
Fs = newSamplingRate;

% Calculate harmonic ratios
HRatio_ML = calculateHarmonicRatio(ML_segment, Fs, 0); % ML axis, odd/even
HRatio_AP = calculateHarmonicRatio(AP_segment, Fs, 1); % AP axis, even/odd
HRatio_Vert = calculateHarmonicRatio(Vert_segment, Fs, 1); % Vertical axis, even/odd

% Display results
fprintf('Harmonic Ratio ML: %.4f\n', HRatio_ML);
fprintf('Harmonic Ratio AP: %.4f\n', HRatio_AP);
fprintf('Harmonic Ratio Vertical: %.4f\n', HRatio_Vert);

% Function to calculate the harmonic ratio
function [HRatio] = calculateHarmonicRatio(rAcc, Fs, evenoverodd)
    % Length of signal
    L = length(rAcc);
    
    % Perform FFT on the acceleration signal
    Y = fft(rAcc, L) / L;
    fL = ceil((L+1)/2); % Length of single-sided spectrum
    f = Fs / 2 * linspace(0, 1, fL); % Frequency vector
    amp = 2 * abs(Y(1:fL)); % Single-sided amplitude spectrum

    % Extract harmonic amplitudes
    if length(amp) > 20
        amp_harm = amp(2:21); % Use first 20 harmonics (excluding DC)
    else
        amp_harm = amp(2:end); % Use available harmonics
    end

    % Calculate sum of even and odd harmonics
    if length(amp_harm) > 19
        even_harm = sum(amp_harm(2:2:20));
        odd_harm = sum(amp_harm(1:2:20));
    else
        even_harm = sum(amp_harm(2:2:end));
        odd_harm = sum(amp_harm(1:2:end));
    end

    % Calculate harmonic ratio
    if evenoverodd == 1
        HRatio = even_harm / odd_harm; % For vertical and anteroposterior
    elseif evenoverodd == 0
        HRatio = odd_harm / even_harm; % For mediolateral
    end
end

% Define time vector
time_per_sample = 1 / newSamplingRate;

% Extract segments for stand, turn, and sit
stand_segment = T_labeled.ML_denoised(newStand_start:newStand_end);
turn_segment = T_labeled.ML_denoised(turn_start_global:turn_end_global);
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
% Define the file path for the CSV file
csvFilePath = '/Users/bsuffoletto/Desktop/Accel/UPMC/predictors/predictors.csv';

% Check if the file already exists
fileExists = isfile(csvFilePath);

% Open the CSV file
if ~fileExists
    % If the file does not exist, create it and open in write mode
    fileID = fopen(csvFilePath, 'w');
else
    % If the file exists, open it in append mode
    fileID = fopen(csvFilePath, 'a');
end

% Define the header and data to write
header = {'File Identifier', 'mean_step_time_in_seconds', 'mean_left_stride_time_in_seconds', ...
    'mean_right_stride_time_in_seconds', 'mean_combined_stride_time_in_seconds', ...
    'left_stride_time_std', 'right_stride_time_std', 'combined_stride_time_std', ...
    'mean_left_stance_time_in_seconds', 'mean_right_stance_time_in_seconds', ...
    'mean_combined_stance_time_in_seconds', 'left_stance_time_std', ...
    'right_stance_time_std', 'combined_stance_time_std', ...
    'mean_left_swing_time_in_seconds', 'left_swing_time_std', ...
    'right_swing_time_std', 'combined_swing_time_std', ...
    'mean_right_swing_time_in_seconds', 'mean_combined_swing_time_in_seconds', ...
    'mean_double_support_time_in_seconds', 'std_double_support_time_in_seconds', ...
    'stride_time_asymmetry', 'swing_time_asymmetry', 'stance_time_asymmetry', ...
    'LyE', 'HRatio_AP', 'HRatio_ML', 'HRatio_Vert', 'CI', ...
    'stand_MA', 'turn_MA', 'sit_MA', 'stand_MV', 'turn_MV', 'sit_MV', ...
    'stand_MP', 'turn_MP', 'sit_MP', 'stand_time_to_complete', ...
    'turn_time_to_complete', 'sit_time_to_complete'};

% Example data (replace these with your actual calculated variables)
% Round each numeric value to 5 decimal places
data = {file_identifier, round(mean_step_time_in_seconds, 5), round(mean_left_stride_time_in_seconds, 5), ...
    round(mean_right_stride_time_in_seconds, 5), round(mean_combined_stride_time_in_seconds, 5), ...
    round(left_stride_time_std, 5), round(right_stride_time_std, 5), round(combined_stride_time_std, 5), ...
    round(mean_left_stance_time_in_seconds, 5), round(mean_right_stance_time_in_seconds, 5), ...
    round(mean_combined_stance_time_in_seconds, 5), round(left_stance_time_std, 5), ...
    round(right_stance_time_std, 5), round(combined_stance_time_std, 5), ...
    round(mean_left_swing_time_in_seconds, 5), round(left_swing_time_std, 5), ...
    round(right_swing_time_std, 5), round(combined_swing_time_std, 5), ...
    round(mean_right_swing_time_in_seconds, 5), round(mean_combined_swing_time_in_seconds, 5), ...
    round(mean_double_support_time_in_seconds, 5), round(std_double_support_time_in_seconds, 5), ...
    round(stride_time_asymmetry, 5), round(swing_time_asymmetry, 5), round(stance_time_asymmetry, 5), ...
    round(LyE, 5), round(HRatio_AP, 5), round(HRatio_ML, 5), round(HRatio_Vert, 5), round(CI, 5), ...
    round(stand_MA, 5), round(turn_MA, 5), round(sit_MA, 5), ...
    round(stand_MV, 5), round(turn_MV, 5), round(sit_MV, 5), ...
    round(stand_MP, 5), round(turn_MP, 5), round(sit_MP, 5), ...
    round(stand_time_to_complete, 5), round(turn_time_to_complete, 5), round(sit_time_to_complete, 5)};

% Write the header if the file is being created for the first time
if ~fileExists
    fprintf(fileID, '%s,', header{:});
    fprintf(fileID, '\n');
end

% Write the data row with fixed-point formatting (%.5f)
fprintf(fileID, '%s,', data{1}); % Write file identifier as a string
for i = 2:length(data)
    fprintf(fileID, '%.5f,', data{i}); % Write each numeric value as a fixed-point number
end
fprintf(fileID, '\n');

% Close the file
fclose(fileID);
