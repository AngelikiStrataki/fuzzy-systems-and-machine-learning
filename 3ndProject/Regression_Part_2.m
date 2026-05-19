% This script finds the optimum TSK model for a high dimensioning dataset, by
% using the grid search method. Then the optimum TSK model is training.

% This script performs a grid search to determine the most suitable TSK model for a dataset with many input variables.

% Author: Modified version

%% Clear environment
clearvars;
clc;

%% Load dataset
raw_data = load('superconduct.csv');

%% Preprocess data: split and scale into training, validation, and testing sets
apply_scaling = 1;
[train_set, val_set, test_set] = split_scale(raw_data, apply_scaling);

%% Define hyperparameter options
cluster_radius_list = [0.2, 0.3, 0.4, 0.5, 0.6];
feature_counts = [5, 10, 15, 20, 25];

%% Setup containers and cross-validation
folds = 5;
average_errors = zeros(1, length(feature_counts) * length(cluster_radius_list));
current_index = 1;
cv_errors = zeros(1, folds);

%% Cross-validation configuration
cv_indices = cvpartition(size(train_set,1), 'KFold', folds);
[selected_features, ~] = relieff(train_set(:,1:end-1), train_set(:,end), 10);

%% Grid search across parameters
for f_idx = 1:length(feature_counts)
    for r_idx = 1:length(cluster_radius_list)
        for fold = 1:folds
            tr_data = train_set(cv_indices.training(fold)==1, [selected_features(1:feature_counts(f_idx)), end]);
            val_data = train_set(cv_indices.test(fold)==1, [selected_features(1:feature_counts(f_idx)), end]);

            gen_opts = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', cluster_radius_list(r_idx));
            initial_fis = genfis(tr_data(:,1:end-1), tr_data(:,end), gen_opts);

            [~, ~, ~, ~, valErr] = anfis(tr_data, initial_fis, [30 0 0.01 0.9 1.1], [], val_data);

            cv_errors(fold) = min(valErr);
        end
        average_errors(current_index) = mean(cv_errors);
        current_index = current_index + 1;
    end
end

%% Visualize error surface
[grid_radii, grid_features] = meshgrid(cluster_radius_list, feature_counts);
reshaped_errors = reshape(average_errors, length(cluster_radius_list), length(feature_counts));

figure;
scatter3(grid_radii(:), grid_features(:), reshaped_errors(:), 'filled');
xlabel('Cluster Radius');
ylabel('Selected Features');
zlabel('Validation Error');
title('Grid Search Performance');

%% Determine optimal parameters
[~, best_idx] = min(reshaped_errors(:));
[best_r_idx, best_f_idx] = ind2sub(size(reshaped_errors), best_idx);
best_radius = cluster_radius_list(best_r_idx);
best_feature_count = feature_counts(best_f_idx);

%% Train final model
final_train = train_set(:, [selected_features(1:best_feature_count), end]);
final_val = val_set(:, [selected_features(1:best_feature_count), end]);

opt_opts = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', best_radius);
final_fis = genfis(final_train(:,1:end-1), final_train(:,end), opt_opts);

[trained_fis, trnErr, ~, val_fis, valErr] = anfis(final_train, final_fis, [100 0 0.01 0.9 1.1], [], final_val);

%% Evaluate prediction error
predicted_output = evalfis(val_fis, val_set(:, selected_features(1:best_feature_count)));
error_vector = val_set(:, end) - predicted_output;

figure;
plot(error_vector);
title('Prediction Error (Final Model)');
ylabel('Error');
xlabel('Sample Index');

%% Plot learning performance
figure;
plot([trnErr valErr], 'LineWidth', 2); grid on;
xlabel('Epochs');
ylabel('Error');
legend('Training', 'Validation');
title('Learning Curve');

%% Visualize fuzzy membership functions
figure;
sgtitle('Initial Membership Functions');
for i = 1:best_feature_count
    [x_vals, mfs] = plotmf(final_fis, 'input', i);
    subplot(5,5,i);
    plot(x_vals, mfs);
    xlabel(sprintf('Input %d', i));
end

figure;
sgtitle('Trained Membership Functions');
for i = 1:best_feature_count
    [x_vals, mfs] = plotmf(trained_fis, 'input', i);
    subplot(5,5,i);
    plot(x_vals, mfs);
    xlabel(sprintf('Input %d', i));
end

%% Compute performance metrics
MSE = mean(error_vector.^2);
RMSE = sqrt(MSE);
R_squared = 1 - sum((predicted_output - val_set(:,end)).^2) / sum((val_set(:,end) - mean(val_set(:,end))).^2);
NMSE = 1 - R_squared;
NDEI = sqrt(NMSE);

fprintf('\n============================================================\n');
fprintf('Final Model: RMSE = %f  NMSE = %f  NDEI = %f  R^2 = %f\n', RMSE, NMSE, NDEI, R_squared);
fprintf('============================================================\n');

%% clear 
clearvars;
clc;
%% Load the data from .csv file
data=load('superconduct.csv');

%% Split and normalize the data into three subsets by using the split scale function
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

%% Initialize the possible values for parameters
R_values = [0.2 ,0.3, 0.4, 0.5, 0.6]; 
features_number = [5, 10, 15, 20, 25];       

%% Local-Helpful variables
Num_of_folds = 5;                                   % The number of created folds on the training data
mean_errors = zeros(1,(length(features_number)*length(R_values)));    % An array to save each mean error 
count = 1;
fold_mean_erros = zeros(1,Num_of_folds);            % Create an array to save the mean errors for each fold 

%% Grid Search method
% 5-fold cross validation
cv = cvpartition(size(trnData,1), 'KFold', Num_of_folds); 
% Relief algorithm: Select specific features_number data from 83 total features
[Index,~] = relieff(trnData(:,1:end-1),trnData(:,end),10);

% 2 stage for loop for all the combinations of parameters' values
for i = 1:length(features_number)                       % For each features_number value
    for j = 1:length(R_values)                          % For each R_values
        %fold_mean_erros = zeros(Num_of_folds);         % Create an array to save the mean errors for each fold 
        
        for k = 1:Num_of_folds                          % For each fold
            % Take the training and the validation data for each fold, by
            % using the created indexes from relieff algorithm
            fold_trnData = trnData(cv.training(k) == 1, [Index(1:features_number(i)), end]);
            fold_chkData = trnData(cv.test(k) == 1, [Index(1:features_number(i)), end]);
            
            % Create the fis by using the fold data
            Options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', R_values(j));
            fis = genfis(fold_trnData(:,1:(end-1)), fold_trnData(:,end), Options);

            % Train the fis and calculate the error
            [~,~,~,~,valError]=anfis(fold_trnData,fis,[30 0 0.01 0.9 1.1],[],fold_chkData);  
    
            % Calculate the mean error for each fold
            fold_mean_erros(k) = min(valError);
        end
        % Calculate and save the mean error for this set of parameters
        mean_errors(count) = mean(fold_mean_erros); 
        count = count + 1;
    end
end

%% Plot the 3-D graph between mean_errors, features_number and R_values
% Create a grid of (r_a, feature_numbers) combinations
[R, Features] = meshgrid(R_values, features_number);

% Reshape the error array to match the grid
Errors_Reshaped = reshape(mean_errors, numel(R_values), numel(features_number));

% Create a 3D scatter plot
figure;
scatter3(R(:), Features(:), Errors_Reshaped(:), 'filled');
xlabel('Radius of clusters(r_a)');
ylabel('Number of Features');
zlabel('Mean Error');
title('Relationship between Radius, Number of Features and Mean Error');

%% Find the optimum TSK model
[min_error, min_error_index] = min(Errors_Reshaped(:));
[optimum_r_a_index, optimum_feature_number_index] = ind2sub(size(Errors_Reshaped), min_error_index);
optimum_radius = R_values(optimum_r_a_index);
optimum_feature_number = features_number(optimum_feature_number_index);

%% Train the optimum TSK model
% Create the datasets by using the Indexes fro, the relieff algorithm
optimum_trnData = trnData(:, [Index(1:optimum_feature_number), end]);
optimum_chkData = chkData(:, [Index(1:optimum_feature_number), end]);
            
% Create the fis by using the above data
Options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', optimum_radius);
optimum_fis = genfis(optimum_trnData(:,1:(end-1)), optimum_trnData(:,end), Options);

% Train the fis and calculate the error
[trnFis,trnError,~,valFis,valError]=anfis(optimum_trnData,optimum_fis,[100 0 0.01 0.9 1.1],[],optimum_chkData);  

%% Plot the prediction error for the optimum model 
% Calculate the output of the validation fis and by using the chkData subset
Y = evalfis(valFis, chkData(:,Index(1:optimum_feature_number)));    
    
% Calculate the prediction error
optimum_error = chkData(:,end) - Y;
    
% Create the prediction error plot
figure; 
plot(optimum_error);
title('Optimum TSK model: Prediction Error');
ylabel('Error');
xlabel('Sample index');

%% Plot the learning curves of the optimum model
figure;
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); 
ylabel('Error');
legend('Training Error','Validation Error');
title('Optimum TSK model: Learning Curves');

%% Plot the fuzzy sets before training for the optimum model
figure;
sgtitle(sprintf('TSK membership functions before training'));

for i = 1:optimum_feature_number
    [x,mf] = plotmf(optimum_fis,'input',i);
    subplot(5,5,i);
    plot(x,mf);
    xlabel(sprintf('Membership Functions for Input %d',i));
end

%% Plot the fuzzy sets after training for the optimum model
figure;
sgtitle(sprintf('TSK membership functions after training'));
for i = 1:optimum_feature_number
    [x,mf] = plotmf(trnFis,'input',i);
    subplot(5,5,i);
    plot(x,mf);
    xlabel(sprintf('Membership Functions for Input %d',i));
end

%% Calculate the evalution parameter for the optimum model
%   Mean Square Error (MSE)
MSE = mean(optimum_error.^2);
    
% Root Mean Square Error (RMSE)
RMSE = sqrt(MSE);
    
% Coefficient of determination factor (R^2)
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);   % Evaluation Function
R2=Rsq(Y,chkData(:,end));
    
% Normalized Mean Squared Error (NMSE)
NMSE = 1 - R2;
    
% NDEI factor
NDEI = sqrt(NMSE);

% Print all the factors for each training
fprintf('\n==================================================================================\n');
fprintf('Optimum TSK Model: RMSE = %f  NMSE = %f  NDEI = %f  R2 = %f\n', RMSE, NMSE, NDEI, R2);
fprintf('==================================================================================\n');

