function results = maxexp_eval(eval_params)
% [rewards, accuracies, height_portion, height_acc] = maxrew_eval(eval_params)
% Evaluates a hierarchical classifier learned by MAX-EXP.
% Arguments:
%   eval_params: Struct of parameters.  Specifically, it should contain:
%     decision_values: num_examples x num_leaves matrix of decision values
%       that come from a flat classifier.
%     labels: Vector of length num_examples giving the ground truth label
%       for each example.  Each label should be one of the leaf nodes.
%     thresholds: A vector of learned thresholds.
%     tree: Vector of metadata about the underlying semantic tree.
%     platt_a: Vector of Platt scaling 'a' parameters for each leaf class.
%     platt_b: Vector of Platt scaling 'b' parameters for each leaf class.
% Returns:
%   results: A struct containing the following fields:
%     rewards: A vector of the average reward obtained for each accuracy
%       guarantee.
%     accuracies: A vector of overall accuracies, one for each guarantee.
%     height_portions: A num_heights * num_guarantees matrix of the portion of
%       all predictions at each height, for each accuracy guarantee, in
%       order of increasing height (i.e. starting at the leaves).
%     height_accs: A num_heights * num_guarantees matrix of the accuracy
%       of the predictions at each height, in order of increasing height.

% Unpack parameters
validate_params(eval_params);
labels = eval_params.labels;
tree = eval_params.tree;
thresholds = eval_params.thresholds;
decision_values = eval_params.decision_values;
platt_a = eval_params.platt_a;
platt_b = eval_params.platt_b;

% Do Platt scaling.
num_leaves = nnz([tree.height] == 0);
leaf_probs = decision_values;
for i = 1:num_leaves
  class_a = platt_a(i);
  class_b = platt_b(i);
  leaf_probs(:, i) = 1 ./ (1 + exp(class_a * leaf_probs(:, i) + class_b));
end

% Normalize probabilities on the leaves in order to have a valid distribution.
leaf_probs = bsxfun(@rdivide, leaf_probs, sum(leaf_probs, 2));

tree_rewards = info_rewards(tree);
normed_rewards = tree_rewards ./ max(tree_rewards); % In [0,1]
num_heights = numel(unique([tree.height]));
rewards = zeros(size(thresholds));
accuracies = zeros(size(thresholds));
height_portions = zeros(num_heights, numel(thresholds));
height_accs = zeros(num_heights, numel(thresholds));

% Evaluate on each given threshold.
for i = 1:numel(thresholds)
  % Make predictions and evaluate the reward. 
  preds = maxexp_predict(leaf_probs, tree_rewards, tree, thresholds(i));
  [reward, acc, height_portion, height_acc] = eval_reward(preds, labels, ...
    normed_rewards, tree);
  % Log results
  rewards(i) = reward;
  accuracies(i) = acc;
  height_portions(:, i) = height_portion(:);
  height_accs(:, i) = height_acc(:);
end
results.rewards = rewards;
results.accuracies = accuracies;
results.height_portions = height_portions;
results.height_accs = height_accs;
end

function validate_params(params)
  % Makes sure everything in the given options exists.
  assert(isfield(params, 'thresholds'));
  assert(isfield(params, 'labels'));
  assert(isfield(params, 'tree'));
  assert(isfield(params, 'decision_values'));
  assert(isfield(params, 'platt_a'));
  assert(isfield(params, 'platt_b'));
end
