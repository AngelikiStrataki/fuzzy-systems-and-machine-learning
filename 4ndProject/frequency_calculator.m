% Calculate the class frequency in each subset (train, check, test)

% --- Training set ---
train_class = train_data(:, 4);

% Count class occurrences
count_train_1 = sum(train_class == 1);
count_train_2 = sum(train_class == 2);

% Total instances
num_train = numel(train_class);

% Frequency in percentage
freq_train_1 = 100 * (count_train_1 / num_train);
freq_train_2 = 100 * (count_train_2 / num_train);


% --- Validation (check) set ---
check_class = check_data(:, 4);

% Count class occurrences
count_check_1 = sum(check_class == 1);
count_check_2 = sum(check_class == 2);

% Total instances
num_check = numel(check_class);

% Frequency in percentage
freq_check_1 = 100 * (count_check_1 / num_check);
freq_check_2 = 100 * (count_check_2 / num_check);


% --- Test set ---
test_class = test_data(:, 4);

% Count class occurrences
count_test_1 = sum(test_class == 1);
count_test_2 = sum(test_class == 2);

% Total instances
num_test = numel(test_class);

% Frequency in percentage
freq_test_1 = 100 * (count_test_1 / num_test);
freq_test_2 = 100 * (count_test_2 / num_test);


% --- Display results ---
fprintf('Frequency in training data:\n');
fprintf('  Class 1: %.2f%%\n', freq_train_1);
fprintf('  Class 2: %.2f%%\n', freq_train_2);

fprintf('Frequency in check data:\n');
fprintf('  Class 1: %.2f%%\n', freq_check_1);
fprintf('  Class 2: %.2f%%\n', freq_check_2);

fprintf('Frequency in test data:\n');
fprintf('  Class 1: %.2f%%\n', freq_test_1);
fprintf('  Class 2: %.2f%%\n', freq_test_2);
