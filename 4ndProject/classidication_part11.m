% TSK classification using Haberman's Survival dataset with Subtractive Clustering

% Clear environment
clear;
clc;

% Load and sort dataset
load haberman.data;
raw_data = haberman;
sorted_data = sortrows(raw_data, 4);

% Split and normalize data into subsets
preproc = 1;
[train_data, check_data, test_data] = split_scale(sorted_data, preproc);

% Model configurations: Class Independent
% Model A - smaller influence range
options(1) = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.2);
% Model B - larger influence range
options(2) = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.8);

% Loop through both model configurations
for idx = 1:2
    % Generate initial FIS
    fis_initial = genfis(train_data(:, 1:end-1), train_data(:, end), options(idx));
    
    % Train FIS using ANFIS
    [trained_fis, train_error, ~, validated_fis, valid_error] = ...
        anfis(train_data, fis_initial, [100 0 0.01 0.9 1.1], [], check_data);

    % Predict and round outputs
    predicted_labels = round(evalfis(test_data(:, 1:end-1), validated_fis));
    
    % Plot membership functions post-training
    figure;
    sgtitle(sprintf('Membership Functions - Model %d', idx));
    for inp_idx = 1:3
        subplot(3, 3, inp_idx);
        [x_vals, mf_vals] = plotmf(trained_fis, 'input', inp_idx);
        plot(x_vals, mf_vals);
        xlabel(sprintf('Input %d', inp_idx));
        title('MFs');
    end

    % Plot learning curves
    figure;
    plot([train_error valid_error], 'LineWidth', 2);
    grid on;
    xlabel('Epochs');
    ylabel('Error');
    legend('Training', 'Validation');
    title(sprintf('Learning Curve - Model %d', idx));
    
    % Evaluation Metrics
    test_class = test_data(:, end);
    unique_classes = unique(raw_data(:, end));
    confusion_mat = zeros(length(unique_classes));
    
    for row = 1:length(test_data)
        pred_idx = find(unique_classes == predicted_labels(row));
        true_idx = find(unique_classes == test_class(row));
        if ~isempty(pred_idx) && ~isempty(true_idx)
            confusion_mat(pred_idx, true_idx) = confusion_mat(pred_idx, true_idx) + 1;
        end
    end

    % Overall Accuracy
    OA = trace(confusion_mat) / length(test_data);
    
    % Producer’s Accuracy
    true_total = sum(confusion_mat, 1);
    for c = 1:length(true_total)
        PA(c) = confusion_mat(c, c) / true_total(c);
    end

    % User’s Accuracy
    pred_total = sum(confusion_mat, 2);
    for c = 1:length(pred_total)
        UA(c) = confusion_mat(c, c) / pred_total(c);
    end

    % Kappa Statistic
    kappa_numerator = length(test_data) * trace(confusion_mat) - sum(PA .* UA) * length(test_data);
    kappa_denominator = length(test_data)^2 - sum(PA .* UA) * length(test_data);
    K = kappa_numerator / kappa_denominator;

    % Display results
    fprintf('\n=========================================================\n');
    fprintf('Model %d Results:\n', idx);
    disp('Confusion Matrix:');
    disp(confusion_mat);
    fprintf('Overall Accuracy (OA): %.3f\n', OA);
    fprintf('Producer''s Accuracy (PA): %.3f\n', PA);
    fprintf('User''s Accuracy (UA): %.3f\n', UA);
    fprintf('Kappa (K̂): %.3f\n', K);
    fprintf('=========================================================\n');
end
