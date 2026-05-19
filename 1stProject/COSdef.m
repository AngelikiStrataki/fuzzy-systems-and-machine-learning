function y_defuzz = COSdef(u_vals, m_vals)

    % u_vals: τιμές x της MF (μεταβλητής εξόδου)
    % m_vals: βαθμοί συμμετοχής

    [peak_heights, peak_positions] = findpeaks(m_vals, u_vals); % Εύρεση τοπικών μέγιστων

    area_weighted_sum = 0;
    area_sum = 0;

    for k = 1:length(peak_heights)
        start_idx = find(u_vals == peak_positions(k));
        mf_width = length(m_vals(m_vals == m_vals(start_idx)));
        end_idx = start_idx + mf_width - 1;

        base = abs(u_vals(end_idx) - u_vals(start_idx));
        region_area = 0.5 * (1/4 + base) * peak_heights(k);
        center = 0.5 * (u_vals(end_idx) + u_vals(start_idx));

        area_sum = area_sum + region_area;
        area_weighted_sum = area_weighted_sum + region_area * center;
    end

    % Αντιμετώπιση άκρου NV
    if m_vals(1) ~= 0
        i_start = 1;
        width = length(m_vals(m_vals == m_vals(i_start)));
        i_end = i_start + width - 1;

        base = abs(u_vals(i_end) - u_vals(i_start));
        reg_area = 0.5 * (1/4 + base) * m_vals(1);

        f1 = @(x) m_vals(1) .* x;
        f2 = @(x) ((m_vals(1) / (u_vals(i_end) + 0.5)) .* (x + 0.5) .* x);

        I1 = integral(f1, -1, u_vals(i_end));
        I2 = integral(f2, u_vals(i_end), -0.5);
        I_total = I1 + I2;

        reg_center = I_total / reg_area;
        area_sum = area_sum + reg_area;
        area_weighted_sum = area_weighted_sum + reg_center * reg_area;
    end

    % Αντιμετώπιση άκρου PV
    if m_vals(end) ~= 0
        i_start = 101;
        width = length(m_vals(m_vals == m_vals(i_start)));
        i_end = i_start - width - 1;

        base = abs(u_vals(i_start) - u_vals(i_end));
        reg_area = 0.5 * (1/4 + base) * m_vals(end);

        f1 = @(x) ((m_vals(end) / (u_vals(i_end) - 0.5)) .* (x - 0.5) .* x);
        f2 = @(x) m_vals(end) .* x;

        I1 = integral(f1, 0.5, u_vals(i_end));
        I2 = integral(f2, u_vals(i_end), 1);
        I_total = I1 + I2;

        reg_center = I_total / reg_area;
        area_sum = area_sum + reg_area;
        area_weighted_sum = area_weighted_sum + reg_center * reg_area;
    end

    y_defuzz = area_weighted_sum / area_sum;

end
