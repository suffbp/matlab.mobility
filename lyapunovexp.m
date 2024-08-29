%% From: Amirpourabasi A, Lamb SE, Chow JY, Williams GKR. Nonlinear Dynamic Measures of Walking in Healthy Older Adults: A Systematic Scoping Review. Sensors (Basel). 2022 Jun 10;22(12):4408. doi: 10.3390/s22124408.% Define necessary parameters
newSamplingRate = 50; % Adjust based on your data
segment_indices = newStand_end:newSit_start; % Indices for walking segment

% Extract walking segment from T_labeled
walking_segment = T_labeled.ML_denoised(segment_indices); % Use Mediolateral axis for example

% Calculate Lyapunov Exponent
LyE = calculateLyapunovExponent(walking_segment, newSamplingRate);

% Display the result
fprintf('Lyapunov Exponent for walking segment: %.4f\n', LyE);

% Plot the walking segment and show Lyapunov Exponent
figure;
plot(walking_segment, 'LineWidth', 1.5);
title('Walking Segment');
xlabel('Sample Index');
ylabel('Acceleration (ML)');
grid on;
text(length(walking_segment) * 0.8, max(walking_segment) * 0.8, sprintf('LyE = %.4f', LyE), 'FontSize', 12, 'Color', 'r');

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
