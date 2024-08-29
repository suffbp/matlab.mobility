function [pslt, pslh, psrt, psrh] = strdesegm(T_labeled, newStand_end, newSit_start, turn_start_global, turn_end_global, newSamplingRate)

    % Define the segment of interest excluding the turn section
    segment_indices = [newStand_end:turn_start_global-1, turn_end_global+1:newSit_start];

    % Get AP, ML, and Vert data segments excluding the turn section
    x = T_labeled.ML_denoised(segment_indices);  % ML data (Medial-Lateral)
    y = T_labeled.Vert_denoised(segment_indices); % Vert data (Vertical)
    z = T_labeled.AP_denoised(segment_indices);  % AP data (Anterior-Posterior)

    % Normalize the data
    x = x / max(abs(x));
    y = y / max(abs(y));
    z = z / max(abs(z));

    % Set parameters
    numbpx = 7; % Number of points to check for left/right distinction
    stdist = round(0.4 * newSamplingRate); % Minimum peak distance based on sampling frequency

    % Apply median filtering
    xp = medfilt1(x - mean(x), 5);
    zp = medfilt1(z - mean(z), 5);
    yp = medfilt1(y - mean(y), 5);

    % Detect peaks in AP (z) component
    [ppks, ploc] = findpeaks(zp, 'minpeakdistance', stdist);
    [npks, nloc] = findpeaks(-zp, 'minpeakdistance', stdist);
    npks = -npks;

    % Check for too few peaks
    if length(ploc) < 3 || length(nloc) < 3
        pslt = [];
        pslh = [];
        psrt = [];
        psrh = [];
        return;
    end

    % Remove initial peaks if too close to the start
    if ploc(1) < 0.3 * newSamplingRate
        ploc = ploc(2:end);
        ppks = ppks(2:end);
    end
    if nloc(1) < 0.3 * newSamplingRate
        nloc = nloc(2:end);
        npks = npks(2:end);
    end

    % Adjust abnormal peak distances in nloc
    dnloc = diff(nloc);
    maxstd = find(dnloc > (mean(dnloc) + 2 * std(dnloc))) + 1;
    for kk = 1:length(maxstd)
        if maxstd(kk) < (length(nloc) - 1)
            nloc(maxstd(kk)) = nloc(maxstd(kk) - 1) + round((nloc(maxstd(kk) + 1) - nloc(maxstd(kk) - 1)) / 2);
        end
    end
    minstd = find(dnloc < (mean(dnloc) - 2 * std(dnloc))) + 1;
    for kk = 1:length(minstd)
        if minstd(kk) < (length(nloc) - 1)
            nloc(minstd(kk)) = nloc(minstd(kk) - 1) + round((nloc(minstd(kk) + 1) - nloc(minstd(kk) - 1)) / 2);
        end
    end

    % Adjust abnormal peak distances in ploc
    dploc = diff(ploc);
    pmaxstd = find(dploc > (mean(dploc) + 2 * std(dploc))) + 1;
    if length(pmaxstd) > 1
        for kk = 1:(length(pmaxstd) - 1)
            ploc(pmaxstd(kk)) = ploc(pmaxstd(kk) - 1) + round((ploc(pmaxstd(kk) + 1) - ploc(pmaxstd(kk) - 1)) / 2);
        end
    else
        ploc(pmaxstd) = ploc(pmaxstd - 1) + mean(ploc(1:(pmaxstd - 1)));
    end
    pminstd = find(dploc < (mean(dploc) - 2 * std(dploc))) + 1;
    for kk = 1:length(pminstd)
        if pminstd(kk) < (length(ploc) - 1)
            ploc(pminstd(kk)) = ploc(pminstd(kk) - 1) + round((ploc(pminstd(kk) + 1) - ploc(pminstd(kk) - 1)) / 2);
        end
    end

    % Detect peaks in Vert (y) component for toe-off detection
    [pypks, pyloc] = findpeaks(yp, 'minpeakdistance', stdist);
    if pyloc(1) < 0.3 * newSamplingRate
        pyloc = pyloc(2:end);
        pypks = pypks(2:end);
    end

    if length(pyloc) < 3
        pslt = [];
        pslh = [];
        psrt = [];
        psrh = [];
        return;
    end

    % Adjust peak distances in pyloc
    dpyloc = diff(pyloc);
    pymaxstd = find(dpyloc > (mean(dpyloc) + 2 * std(dpyloc))) + 1;
    for kk = 1:length(pymaxstd)
        if pymaxstd(kk) < (length(pyloc) - 1)
            pyloc(pymaxstd(kk)) = pyloc(pymaxstd(kk) - 1) + round((pyloc(pymaxstd(kk) + 1) - pyloc(pymaxstd(kk) - 1)) / 2);
        end
    end
    pyminstd = find(dpyloc < (mean(dpyloc) - 2 * std(dpyloc))) + 1;
    for kk = 1:length(pyminstd)
        if pyminstd(kk) < (length(pyloc) - 1)
            pyloc(pyminstd(kk)) = pyloc(pyminstd(kk) - 1) + round((pyloc(pyminstd(kk) + 1) - pyloc(pyminstd(kk) - 1)) / 2);
        end
    end

    % Calculate heel-strike and toe-off events based on peaks
    addcnst = round(0.3 * newSamplingRate);
    Lyp = length(yp);
    if (Lyp - pyloc(end)) < addcnst
        pyloc = pyloc(1:(length(pyloc) - 1));
    end
    pksloc = zeros(1, length(pyloc));
    for kk = 1:length(pyloc)
        temploc = pyloc(kk);
        endpnt = min([(temploc + addcnst), Lyp]);
        tempy = yp(temploc:endpnt);
        [temppks, tempploc] = findpeaks(-tempy, 'npeaks', 1);
        if size(temppks, 1) < 1
            [temppks, tempploc] = min(tempy);
        end
        pksloc(kk) = temploc + tempploc;
    end

    if mean(xp(1:numbpx)) > 0
        psrt = pksloc(2:2:end);
        pslt = pksloc(1:2:end);
    else
        psrt = pksloc(1:2:end);
        pslt = pksloc(2:2:end);
    end

    % Detect heel-strike events
    Nloc = [];
    heelout = 0.8 * newSamplingRate;
    healoffset = 0.07 * newSamplingRate;
    for pp = 1:(length(nloc) - 1)
        if (nloc(pp + 1) - nloc(pp)) < heelout
            Nloc = [Nloc, nloc(pp)];
        end
    end
    Nloc = [Nloc, nloc(end)];

    Lzp = length(zp);
    if (Lzp - nloc(end)) < addcnst
        Nloc = Nloc(1:(length(Nloc) - 1));
    end

    if mean(xp(1:numbpx)) > 0
        psrh = Nloc(1:2:end);
        pslh = Nloc(2:2:end);
    else
        psrh = Nloc(2:2:end);
        pslh = Nloc(1:2:end);
    end

    % Final adjustments to ensure consistency
    mxleng = min([length(pslh), length(psrh), length(psrt), length(pslh)]);
    pslt = pslt(1:mxleng);
    pslh = pslh(1:mxleng);
    psrt = psrt(1:mxleng);
    psrh = psrh(1:mxleng);
end
