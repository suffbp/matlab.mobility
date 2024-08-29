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

% Plot the MSE results
figure;
plot(1:max_scale, mse_values_cleaned, '-o', 'LineWidth', 1.5);
title('Multiscale Entropy (MSE)');
xlabel('Scale');
ylabel('Sample Entropy');
grid on;

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
