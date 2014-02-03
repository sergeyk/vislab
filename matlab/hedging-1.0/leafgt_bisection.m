function thresholds = leafgt_bisection(bs_params)
% thresholds = leafgt_bisection(bs_params)
% Learns the thresholds for LEAF-GT via bisection.
% Arguments:
%   bs_params: Struct of parameters.  Specifically, it should contain:
%     accuracy_guarantees: Vector of accuracy guarantees desired, in [0,1].
%     decision_values: num_examples x num_leaves matrix of decision values
%       that come from a flat classifier.
%     labels: Vector of length num_examples giving the ground truth label
%       for each example.  Each label should be one of the leaf nodes.
%     tree: Vector of metadata about the underlying semantic tree.
%     num_iters: Number of iterations to do binary search for.
%     confidence: The extent of our one-sided confidence interval about the
%       accuracy guarantee.  e.g. if 'confidence' is .95 then the binary
%       search will attempt to find a threshold such that a 95% confidence
%       interval around the observed accuracy is above the accuracy guarantee.
% Returns:
%   thresholds: A vector of thresholds found via bisection, one for each
%     accuracy guarantee given.  The thresholds are relative to the decision
%     values for each node.

% Unpack parameters
validate_params(bs_params);
accuracy_guarantees = bs_params.accuracy_guarantees;
decision_values = bs_params.decision_values;
labels = bs_params.labels;
tree = bs_params.tree;
num_bs_iters = bs_params.num_iters;
confidence = bs_params.confidence;

% Basic parameter parsing
num_leaves = nnz([tree.height] == 0);
num_examples = numel(labels);
[~, root_index] = max([tree.height]);
assert(size(decision_values, 2) == num_leaves);
rewards = info_rewards(tree);

% Get the base predictions, equivalent to a -Inf threshold.
[max_dec_values, flat_predictions] = max(decision_values, [], 2);

% Find the best threshold to use for each accuracy guarantee.
thresholds = zeros(numel(accuracy_guarantees), 1);
for i = 1:numel(accuracy_guarantees)
  guarantee = accuracy_guarantees(i);
  epsilon = 1 - guarantee;

  % 'alpha' corresponds to the p-value we have to match in order to be
  % confident we are above the accuracy guarantee.
  % We can double alpha because we only care about a 1-sided confidence
  % interval, but binofit returns a 2-sided confidence interval.
  desired_alpha = (1 - confidence) * 2;

  % Find the value of threshold to use via binary search.
  % These are reasonable bounds on decision values found via an SVM.
  min_threshold = -100;
  max_threshold = 100;
  for j = 1:num_bs_iters
    current_threshold = (min_threshold + max_threshold) / 2;
    % Make predictions and evaluate the reward. 
    use_flat_predictions = max_dec_values > current_threshold;
    preds = use_flat_predictions .* flat_predictions + ...
      ~use_flat_predictions * root_index;
    [reward, accuracy] = eval_reward(preds, labels, rewards, tree);
    % Maintain a confidence interval around the target accuracy.
    [~, acc_bounds] = ...
      binofit(accuracy * num_examples, num_examples, desired_alpha);
    acc_lower_bound = acc_bounds(1);
    if acc_lower_bound > guarantee
      max_threshold = current_threshold;
    else
      min_threshold = current_threshold;
    end
  end
  % The current upper bound on the optimal threshold is our tightest bound.
  thresholds(i) = max_threshold;
end
end


function validate_params(params)
  % Makes sure everything in the given parameters exists.
  assert(isfield(params, 'accuracy_guarantees'));
  assert(isfield(params, 'decision_values'));
  assert(isfield(params, 'labels'));
  assert(isfield(params, 'tree'));
  assert(isfield(params, 'num_iters'));
  assert(isfield(params, 'confidence'));
end
