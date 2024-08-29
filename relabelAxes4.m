function [mlAxis, apAxis, vertAxis] = relabelAxes4(T_resampled)
    % Extract the filtered smoothed data from the table
    X = T_resampled.X_smoothed;
    Y = T_resampled.Y_smoothed;
    Z = T_resampled.Z_smoothed;

    % Calculate the mean absolute value difference from zero for each axis
    meanDiffFromZero = [mean(abs(X)), mean(abs(Y)), mean(abs(Z))];

    % Calculate the dynamic range (max - min) for each axis
    dynamicRange = [range(X), range(Y), range(Z)];

    % Combine the metrics for a better classification
    combinedMetric = meanDiffFromZero .* dynamicRange;

    % Find the axis with the largest combined metric and assign it as the Vertical axis
    [~, verticalIdx] = max(combinedMetric);
    axesNames = {'X', 'Y', 'Z'};
    verticalAxisName = axesNames{verticalIdx};

    % Calculate the maximum peak for each axis in the first third of the data series
    dataLength = length(X);
    oneThirdLength = floor(dataLength / 3);
    peakValues = [max(abs(X(1:oneThirdLength))), max(abs(Y(1:oneThirdLength))), max(abs(Z(1:oneThirdLength)))];

    % Exclude the Vertical axis and find the axis with the largest initial peak
    peakValues(verticalIdx) = 0; % Exclude the vertical axis
    [~, apAxisIdx] = max(peakValues);
    apAxisName = axesNames{apAxisIdx};

    % The remaining axis is the ML axis
    mlAxisIdx = setdiff(1:3, [verticalIdx, apAxisIdx]);
    mlAxisName = axesNames{mlAxisIdx};

    % Assign the axis data to new variables
    mlAxis = T_resampled.([mlAxisName, '_smoothed']);
    apAxis = T_resampled.([apAxisName, '_smoothed']);
    vertAxis = T_resampled.([verticalAxisName, '_smoothed']);

    % Display the assigned axes for verification
    fprintf('Vertical Axis: %s\n', verticalAxisName);
    fprintf('AP Axis: %s\n', apAxisName);
    fprintf('ML Axis: %s\n', mlAxisName);

    % Plot the results
    figure;

    % Plot AP axis data
    h1 = subplot(3, 1, 1);
    plot(apAxis, 'b'); % Smoothed data in blue
    title('AP Axis');
    xlabel('Sample Index');
    ylabel('Acceleration');

    % Plot ML axis data
    h2 = subplot(3, 1, 2);
    plot(mlAxis, 'b'); % Smoothed data in blue
    title('ML Axis');
    xlabel('Sample Index');
    ylabel('Acceleration');

    % Plot Vertical axis data
    h3 = subplot(3, 1, 3);
    plot(vertAxis, 'b'); % Smoothed data in blue
    title('Vertical Axis');
    xlabel('Sample Index');
    ylabel('Acceleration');

    % Link the y-axes of all three subplots
    linkaxes([h1, h2, h3], 'y');
end
