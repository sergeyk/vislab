function preds = maxrew_predict(leaf_probs, rewards, tree, threshold)
% preds = maxrew_predict(leaf_probs, rewards, tree)
% Makes predictions based on maximizing possible rewards subject to a
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
all_probs = get_all_probs(leaf_probs, tree);
% Apply the threshold
all_probs = all_probs .* (all_probs >= threshold);
% Maximize reward
probs_01 = all_probs;
probs_01(probs_01 > 0) = 1;
rew = bsxfun(@times, probs_01, rewards(:)');
% Use the probability as a tiebreaker.
rew = .5 * min(diff(unique(rewards))) * all_probs + rew;
[~, preds] = max(rew, [], 2);
