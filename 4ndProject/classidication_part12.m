% Training TSK models 3 & 4 (Class-dependent Subtractive Clustering)

clear;
clc;

% Load and sort dataset
load haberman.data;
dataset = haberman;
sorted_data = sortrows(dataset, 4);

% Split into train/validate/test
preproc = 1;
[train_data, check_data, test_data] = split_scale(sorted_data, preproc);

% Define radii for SC clustering
radii = [0.2 0.8];

input_names = {'in1', 'in2', 'in3'};

for l = 1:2
    % Class-dependent clustering
    [centers1, sig1] = subclust(train_data(train_data(:, end)==1, :), radii(l));
    [centers2, sig2] = subclust(train_data(train_data(:, end)==2, :), radii(l));
    total_rules = size(centers1, 1) + size(centers2, 1);

    % New FIS
    fis = newfis('TSK_FIS', 'sugeno');

    % Add inputs
    for i = 1:3
        fis = addInput(fis, [0 1], 'Name', input_names{i});
    end

    % Add output
    fis = addOutput(fis, [1 2], 'Name', 'out1');

    % Add input MFs - class 1
    for i = 1:3
        for j = 1:size(centers1, 1)
            mf_name = sprintf('c1_%d', j);
            fis = addMF(fis, input_names{i}, 'gaussmf', [sig1(i), centers1(j, i)], 'Name', mf_name);
        end
        for j = 1:size(centers2, 1)
            mf_name = sprintf('c2_%d', j);
            fis = addMF(fis, input_names{i}, 'gaussmf', [sig2(i), centers2(j, i)], 'Name', mf_name);
        end
    end

    % Add output MFs (constant singleton)
    mf_outputs = [ones(1, size(centers1, 1)), 2*ones(1, size(centers2, 1))];
    for j = 1:total_rules
        mf_name = sprintf('out%d', j);
        fis = addMF(fis, 'out1', 'constant', mf_outputs(j), 'Name', mf_name);
    end

    % Add rules (correct format: [in1 in2 in3 out weight and/or])
    input_rule_indices = repmat((1:total_rules)', 1, 3);
    output_indices = (1:total_rules)';
    weights = ones(total_rules, 1);
    and_method = ones(total_rules, 1);
    rule_list = [input_rule_indices, output_indices, weights, and_method];

    fis = addRule(fis, rule_list);

    % Training
    [train_fis, train_error, ~, val_fis, val_error] = ...
        anfis(train_data, fis, [100 0 0.01 0.9 1.1], [], check_data);

    % Predictions
    predictions = round(evalfis(test_data(:, 1:end-1), val_fis));

    % Plot MFs after training
    figure;
    sgtitle(sprintf('Membership Functions - Model %d', l+2));
    for i = 1:3
        subplot(3, 3, i);
        [x, mf] = plotmf(train_fis, 'input', i);
        plot(x, mf);
        xlabel(sprintf('Input %d', i));
    end

    % Plot learning curves
    figure;
    plot([train_error val_error], 'LineWidth', 2);
    grid on;
    xlabel('Epochs');
    ylabel('Error');
    legend('Training', 'Validation');
    title(sprintf('Learning Curve - Model %d', l+2));

    % Evaluate
    test_class = test_data(:, end);
    classes = unique(dataset(:, end));
    error_matrix = zeros(length(classes));

    for j = 1:length(test_data)
        x = find(classes == predictions(j));
        y = find(classes == test_class(j));
        if ~isempty(x) && ~isempty(y)
            error_matrix(x, y) = error_matrix(x, y) + 1;
        end
    end

    % Overall Accuracy
    OA = trace(error_matrix) / length(test_data);

    % Producer's accuracy
    xjc = sum(error_matrix);
    PA = diag(error_matrix)' ./ xjc;

    % User's accuracy
    xir = sum(error_matrix, 2);
    UA = diag(error_matrix) ./ xir;

    % Kappa coefficient
    sum_prod = sum(PA .* UA);
    K = (length(test_data) * trace(error_matrix) - sum_prod) / ...
        (length(test_data)^2 - sum_prod);

    % Display
    fprintf('\n==================== Model %d ====================\n', l+2);
    disp('Confusion Matrix:');
    disp(error_matrix);
    fprintf('Overall Accuracy (OA): %.3f\n', OA);
    fprintf('Producer''s Accuracy (PA): %.3f %.3f\n', PA);
    fprintf('User''s Accuracy (UA): %.3f %.3f\n', UA);
    fprintf('Kappa (K̂): %.3f\n', K);
    fprintf('===================================================\n');
end
