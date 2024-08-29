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
