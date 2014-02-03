function preds = maxexp_predict(leaf_probs, rewards, tree, threshold)
% preds = maxexp_predict(leaf_probs, rewards, tree)
% Makes predictions based on maximizing expected reward subject to a
% probability threshold.
% Arguments:
%   leaf_probs: num_examples x num_leaves matrix of probabilties of each
%     example at each of the leaf nodes.
%   rewards: The rewards for each node in the tree.
%   tree: Vector of information about the tree.
%   threshold: Threshold to apply to probabilities before predicting.  If a
%     node has a probability below the given threshold, it won't be predicted.
% Returns:
%   preds: Vector of predictions.

num_examples = size(leaf_probs, 1);
num_classes = numel(tree);
root_class = find([tree.height] == max([tree.height]));
all_probs = get_all_probs(leaf_probs, tree);
% Apply the threshold
all_probs = all_probs .* (all_probs >= threshold);
% Maximize expected reward
expected_rewards = bsxfun(@times, all_probs, rewards(:)');
[maxes, preds] = max(expected_rewards, [], 2);
% Sometimes the best reward is 0 if it isn't confident about anything
% (i.e. the threshold is high enough that nothing satisfies it) and
% the root reward is 0, which makes all expected rewards 0, in which
% case max picks the first node instead of the root.
% Detect when this happens and predict the root instead.
preds(maxes == 0) = root_class;
