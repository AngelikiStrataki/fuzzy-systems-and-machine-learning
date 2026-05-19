% TSK Regression Analysis using Airfoil Self-Noise dataset
% This script evaluates four TSK fuzzy inference models using grid partitioning

% Load dataset
load airfoil_self_noise.dat;
data = airfoil_self_noise;

% Normalize and split dataset into training, validation, and testing subsets
normalize_flag = 1;
[train_set, val_set, test_set] = split_scale(data, normalize_flag);

% Configuration for the four fuzzy models
% Model 1: 2 membership functions per input, constant output
fuzzy_config(1) = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, ...
    'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'constant');

% Model 2: 3 membership functions per input, constant output
fuzzy_config(2) = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, ...
    'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'constant');

% Model 3: 2 membership functions per input, linear output
fuzzy_config(3) = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, ...
    'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'linear');

% Model 4: 3 membership functions per input, linear output
fuzzy_config(4) = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, ...
    'InputMembershipFunctionType', 'gbellmf', 'OutputMembershipFunctionType', 'linear');

% Iterate through each TSK model
for m = 1:4

    % Generate initial FIS structure
    initial_fis = genfis(train_set(:, 1:5), train_set(:, 6), fuzzy_config(m));

    % Display input membership functions before training
    figure;
    sgtitle(sprintf('Input Membership Functions Before Training - Model %d', m));
    for j = 1:5
        [x_vals, mf_vals] = plotmf(initial_fis, 'input', j);
        subplot(5, 3, j);
        plot(x_vals, mf_vals);
        xlabel(sprintf('Input %d', j));
    end

    % Train fuzzy model using hybrid method (backpropagation + least squares)
    [trained_fis, train_err, ~, eval_fis, val_err] = anfis(train_set, initial_fis, ...
        [100 0 0.01 0.9 1.1], [], val_set);

    % Display input membership functions after training
    figure;
    sgtitle(sprintf('Input Membership Functions After Training - Model %d', m));
    for j = 1:5
        [x_vals, mf_vals] = plotmf(trained_fis, 'input', j);
        subplot(5, 3, j);
        plot(x_vals, mf_vals);
        xlabel(sprintf('Input %d', j));
    end

    % Plot training and validation error over iterations
    figure;
    plot([train_err val_err], 'LineWidth', 2);
    grid on;
    xlabel('Epoch');
    ylabel('Error');
    legend('Training', 'Validation');
    title(sprintf('Training Progress - Model %d', m));

    % Evaluate prediction on test data
    prediction = evalfis(eval_fis, test_set(:,1:5));
    residuals = test_set(:,6) - prediction;

    % Plot prediction residuals
    figure;
    plot(residuals);
    title(sprintf('Prediction Residuals - Model %d', m));
    xlabel('Sample Index');
    ylabel('Residual');

    % Compute performance metrics
    mse_val = mean(residuals.^2);
    rmse_val = sqrt(mse_val);
    ss_res = sum((prediction - test_set(:,6)).^2);
    ss_tot = sum((test_set(:,6) - mean(test_set(:,6))).^2);
    r_squared = 1 - ss_res / ss_tot;
    nmse_val = 1 - r_squared;
    ndei_val = sqrt(nmse_val);

    % Display metrics in command window
    fprintf('\n------------------------------------------------------------\n');
    fprintf('TSK Model %d: MSE = %.4f, RMSE = %.4f, R^2 = %.4f, NMSE = %.4f, NDEI = %.4f\n', ...
        m, mse_val, rmse_val, r_squared, nmse_val, ndei_val);
    fprintf('------------------------------------------------------------\n');
end
