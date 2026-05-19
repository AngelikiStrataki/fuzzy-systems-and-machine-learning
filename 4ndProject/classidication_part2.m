% Grid Search for Optimum TSK Model - Epileptic Seizure Dataset

clear;
clc;

% Load dataset
dataset = csvread('epileptic_seizure_data.csv', 1, 1);

% Split and normalize
preproc = 1;
[train_data, check_data, test_data] = split_scale(dataset, preproc);

% Grid search parameters
features_number = [5, 10, 15, 20, 25]; 
r_a = [0.2, 0.3, 0.4, 0.5, 0.6]; 
folds = 5;
mean_errors = zeros(1, length(features_number) * length(r_a)); 
fold_errors = zeros(1, folds);
cnt = 1;

% Create folds
cvp = cvpartition(size(train_data, 1), 'KFold', folds); 
[selected_indices, ~] = relieff(train_data(:, 1:end-1), train_data(:, end), 10);

% Grid search
for i = 1:length(features_number)
    for j = 1:length(r_a)
        for k = 1:folds
            fold_train = train_data(cvp.training(k), [selected_indices(1:features_number(i)), end]);
            fold_val = train_data(cvp.test(k), [selected_indices(1:features_number(i)), end]);

            sc_options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', r_a(j));
            fis_fold = genfis(fold_train(:, 1:end-1), fold_train(:, end), sc_options);

            [~, ~, ~, ~, val_err] = anfis(fold_train, fis_fold, [30 0 0.01 0.9 1.1], [], fold_val);
            fold_errors(k) = min(val_err);
        end
        mean_errors(cnt) = mean(fold_errors);
        cnt = cnt + 1;
    end
end

% Plot grid search results
[features_grid, r_grid] = meshgrid(features_number, r_a);
errors_reshaped = reshape(mean_errors, numel(r_a), numel(features_number));

figure;
scatter3(r_grid(:), features_grid(:), errors_reshaped(:), 'filled');
xlabel('Cluster Radius r_a'); ylabel('Number of Features'); zlabel('Mean Error');

% Find optimum parameters
[min_err, min_idx] = min(errors_reshaped(:));
[r_idx, f_idx] = ind2sub(size(errors_reshaped), min_idx);
opt_r = r_a(r_idx);
opt_feat = features_number(f_idx);

% Prepare datasets with selected features
train_selected = train_data(:, [selected_indices(1:opt_feat), end]);
check_selected = check_data(:, [selected_indices(1:opt_feat), end]);

% Build and train final FIS
sc_options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', opt_r);
fis_initial = genfis(train_selected(:, 1:end-1), train_selected(:, end), sc_options);

[fis_trained, train_err, ~, fis_validated, val_err] = ...
    anfis(train_selected, fis_initial, [100 0 0.01 0.9 1.1], [], check_selected);

% Evaluate on check_data
predicted_labels = round(evalfis(fis_validated, check_data(:, selected_indices(1:opt_feat))));
prediction_error = check_data(:, end) - predicted_labels;

% Plot prediction error
figure;
plot(prediction_error);
title('Prediction Error of Optimum Model');
xlabel('Sample Index'); ylabel('Error');

% Plot learning curves
figure;
plot([train_err val_err], 'LineWidth', 2);
grid on;
xlabel('Epochs'); ylabel('Error');
legend('Training', 'Validation');
title('Learning Curves - Optimum Model');

% Plot fuzzy sets before training
figure;
sgtitle('Membership Functions Before Training');
a1 = round(sqrt(opt_feat));
a2 = round(opt_feat / a1) + 1;
for j = 1:opt_feat
    [x, mf] = plotmf(fis_initial, 'input', j);
    subplot(a1, a2, j);
    plot(x, mf);
    xlabel(sprintf('Input %d', j));
end

% Plot fuzzy sets after training
figure;
sgtitle('Membership Functions After Training');
for j = 1:opt_feat
    [x, mf] = plotmf(fis_trained, 'input', j);
    subplot(a1, a2, j);
    plot(x, mf);
    xlabel(sprintf('Input %d', j));
end

% Evaluation metrics on test_data
true_classes = test_data(:, end);
classes = unique(dataset(:, end));
conf_matrix = zeros(length(classes));

for i = 1:length(test_data)
    x = find(classes == predicted_labels(i));
    y = find(classes == true_classes(i));
    if ~isempty(x) && ~isempty(y)
        conf_matrix(x, y) = conf_matrix(x, y) + 1;
    end
end

% Overall Accuracy
OA = trace(conf_matrix) / length(test_data);

% Producer's Accuracy
true_class_total = sum(conf_matrix);
PA = diag(conf_matrix)' ./ true_class_total;

% User's Accuracy
predicted_class_total = sum(conf_matrix, 2);
UA = diag(conf_matrix) ./ predicted_class_total;

% Kappa
sum_prod = sum(PA .* UA);
K = (length(test_data) * trace(conf_matrix) - sum_prod) / ...
    (length(test_data)^2 - sum_prod);

% Print results
fprintf('\n================== Optimum TSK Model ==================\n');
disp('Confusion Matrix:');
disp(conf_matrix);
fprintf('Overall Accuracy (OA): %.3f\n', OA);
fprintf('Producer''s Accuracy (PA): %.3f\n', PA);
fprintf('User''s Accuracy (UA): %.3f\n', UA);
fprintf('Kappa (K̂): %.3f\n', K);
fprintf('=======================================================\n');
