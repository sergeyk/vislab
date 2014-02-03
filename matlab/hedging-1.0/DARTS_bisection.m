function lambdas = DARTS_bisection(bs_params)
% lambdas = DARTS_bisection(bs_params)
% Learns the dual parameter used in DARTS via binary search.
% Arguments:
%   bs_params: Struct of parameters.  Specifically, it should contain:
%     accuracy_guarantees: Vector of accuracy guarantees desired, in [0,1].
%     decision_values: num_examples x num_leaves matrix of decision values
%       that come from a flat classifier.
%     labels: Vector of length num_examples giving the ground truth label
%       for each example.  Each label should be one of the leaf nodes.
%     tree: Vector of metadata about the underlying semantic tree.
%     platt_a: Vector of Platt scaling 'a' parameters for each leaf class.
%     platt_b: Vector of Platt scaling 'b' parameters for each leaf class.
%     num_iters: Number of iterations to do binary search for.
%     confidence: The extent of our one-sided confidence interval about the
%       accuracy guarantee.  e.g. if 'confidence' is .95 then the binary
%       search will attempt to find a lambda such that a 95% confidence
%       interval around the observed accuracy is above the accuracy guarantee.
% Returns:
%   lambdas: A vector of dual parameters found via bisection, one for each
%     accuracy guarantee given.

% Unpack parameters
validate_params(bs_params);
accuracy_guarantees = bs_params.accuracy_guarantees;
decision_values = bs_params.decision_values;
labels = bs_params.labels;
tree = bs_params.tree;
platt_a = bs_params.platt_a;
platt_b = bs_params.platt_b;
num_bs_iters = bs_params.num_iters;
confidence = bs_params.confidence;

% Basic parameter parsing
num_leaves = nnz([tree.height] == 0);
num_examples = numel(labels);
assert(size(decision_values, 2) == num_leaves);
rewards = info_rewards(tree);

% Do Platt scaling.
leaf_probs = decision_values;
for i = 1:num_leaves
  class_a = platt_a(i);
  class_b = platt_b(i);
  leaf_probs(:, i) = 1 ./ (1 + exp(class_a * leaf_probs(:, i) + class_b));
end

% Normalize probabilities on the leaves in order to have a valid distribution.
leaf_probs = bsxfun(@rdivide, leaf_probs, sum(leaf_probs, 2));

% Find the best lambda to use for each accuracy guarantee.
lambdas = zeros(numel(accuracy_guarantees), 1);
for i = 1:numel(accuracy_guarantees)
  guarantee = accuracy_guarantees(i);
  epsilon = 1 - guarantee;

  % 'alpha' corresponds to the p-value we have to match in order to be
  % confident we are above the accuracy guarantee.
  % We can double alpha because we only care about a 1-sided confidence
  % interval, but binofit returns a 2-sided confidence interval.
  desired_alpha = (1 - confidence) * 2;

  % Find the value of lambda to use via binary search.
  min_lambda = 0;
  max_lambda = ((1 - epsilon) * max(rewards) - min(rewards)) / epsilon;
  for j = 1:num_bs_iters
    current_lambda = (min_lambda + max_lambda) / 2;
    % Use the transformed rewards when doing the binary search.
    used_rewards = rewards + current_lambda;
    % Make predictions and evaluate the reward. 
    preds = DARTS_predict(leaf_probs, used_rewards, tree);
    [reward, accuracy] = eval_reward(preds, labels, used_rewards, tree);
    % Maintain a confidence interval around the target accuracy.
    [~, acc_bounds] = ...
      binofit(accuracy * num_examples, num_examples, desired_alpha);
    acc_lower_bound = acc_bounds(1);
    if acc_lower_bound > guarantee
      max_lambda = current_lambda;
    else
      min_lambda = current_lambda;
    end
  end
  % The current upper bound on the optimal lambda is our tightest bound.
  lambdas(i) = max_lambda;
end
end


function validate_params(params)
  % Makes sure everything in the given parameters exists.
  assert(isfield(params, 'accuracy_guarantees'));
  assert(isfield(params, 'decision_values'));
  assert(isfield(params, 'labels'));
  assert(isfield(params, 'tree'));
  assert(isfield(params, 'platt_a'));
  assert(isfield(params, 'platt_b'));
  assert(isfield(params, 'num_iters'));
  assert(isfield(params, 'confidence'));
end
