path(path,'third-party/liblinear-1.8/matlab');

meta_location = 'ilsvrc65_meta.mat';

%training feature file is necessary only because we need to load the ids
%of the images for probability calibration. This needs to correspond to
%the model file (model_location)
train_mat_location = '../features/ilsvrc65.train.subset0.llc.mat';
val_mat_location = '../features/ilsvrc65.val.llc.mat';
test_mat_location = '../features/ilsvrc65.test.llc.mat';

%this needs to correspond to the training file (train_mat_location)
model_location = '../models/ilsvrc65.subset0.C100.model.mat'; 

accuracy_guarantees = [0:.1:.8 .85 .9 .95 .99];
num_bs_iters = 25;
confidence = .95; % once we get within the confidence interval around 1-eps,
                  % we stop the binary search

% Compile Platt scaling
if ~exist('plattscaling', 'file')
  fprintf('Compiling Platt scaling...\n');
  mex plattscaling.cpp;
end

% Extract labels, decision values, etc.
load(meta_location);
tree = synsets;

fprintf('Loading gt files...\n');
train_gt_map = read_gt('ilsvrc65.train.gt'); %training ground truth is
                                             %needed for probablity calibration
val_gt_map = read_gt('ilsvrc65.val.gt');
test_gt_map = read_gt('ilsvrc65.test.gt');
fprintf('Loading train\n');
train_data = load(train_mat_location, 'ids'); % 
fprintf('Loading val\n');
val_data = load(val_mat_location);
fprintf('Loading test\n');
test_data = load(test_mat_location);

% Make labels
train_labels = zeros(size(train_data.ids));
val_labels = zeros(size(val_data.ids));
test_labels = zeros(size(test_data.ids));

fprintf('Getting labels...\n');
for i = 1:numel(train_labels)
  train_labels(i) = class2id_map.get(train_gt_map.get(train_data.ids{i}));
end
for i = 1:numel(val_labels)
  val_labels(i) = class2id_map.get(val_gt_map.get(val_data.ids{i}));
end
for i = 1:numel(test_labels)
  test_labels(i) = class2id_map.get(test_gt_map.get(test_data.ids{i}));
end
num_leaves = numel(unique(train_labels));

% Learn Platt scaling parameters
fprintf('Loading model...\n');
load(model_location);
fprintf('Learning Platt scaling params...\n');
[platt_a, platt_b] = learn_platt_params([cv_pred internal_cv_pred], ...
  train_labels, tree);

% Get decision values on val and test
fprintf('Predicting on val\n');
[~, ~, val_dec_values] = predict(val_labels, val_data.betas, model, '-b 0', 'col');
val_dec_values(:, model.Label) = val_dec_values;
fprintf('Predicting on test\n');
[~, ~, test_dec_values] = predict(test_labels, test_data.betas, model, '-b 0', 'col');
test_dec_values(:, model.Label) = test_dec_values;

methods = {};
results = [];

% Run each of the methods.
fprintf('DARTS\n');
methods = [methods 'DARTS'];
% Learn lambdas
lambdas = DARTS_bisection(struct('accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', val_dec_values, 'labels', val_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'num_iters', num_bs_iters, ...
  'confidence', confidence));
% Evaluate on test
test_results = DARTS_eval(...
  struct('accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', test_dec_values, 'labels', test_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'lambdas', lambdas));
results = [results test_results];

fprintf('LEAF-GT\n');
methods = [methods 'LEAF-GT'];
% Learn thresholds
thresholds = leafgt_bisection(struct(...
  'accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', val_dec_values, 'labels', val_labels, 'tree', tree, ...
  'num_iters', num_bs_iters, 'confidence', confidence));
% Evaluate on test
test_results = leafgt_eval(...
  struct('accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', test_dec_values, 'labels', test_labels, 'tree', tree, ...
  'thresholds', thresholds));
results = [results test_results];

fprintf('MAX-REW\n');
methods = [methods 'MAX-REW'];
% Learn thresholds
thresholds = maxrew_bisection(struct(...
  'accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', val_dec_values, 'labels', val_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'num_iters', num_bs_iters, ...
  'confidence', confidence));
test_results = maxrew_eval(...
  struct('accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', test_dec_values, 'labels', test_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'thresholds', thresholds));
results = [results test_results];

fprintf('MAX-EXP\n');
methods = [methods 'MAX-EXP'];
% Learn thresholds
thresholds = maxexp_bisection(struct(...
  'accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', val_dec_values, 'labels', val_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'num_iters', num_bs_iters, ...
  'confidence', confidence));
test_results = maxexp_eval(...
  struct('accuracy_guarantees', accuracy_guarantees, ...
  'decision_values', test_dec_values, 'labels', test_labels, 'tree', tree, ...
  'platt_a', platt_a, 'platt_b', platt_b, 'thresholds', thresholds));
results = [results test_results];


save('inf_results.mat', 'results', 'num_leaves', 'methods', ...
  'accuracy_guarantees');

