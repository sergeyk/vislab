function preds = DARTS_predict(leaf_probs, rewards, tree)
% preds = DARTS_predict(leaf_probs, rewards, tree)
% Makes predictions based on maximizing expected reward.
% Arguments:
%   leaf_probs: num_examples x num_leaves matrix of probabilties of each
%     example at each of the leaf nodes.
%   rewards: The rewards for each node in the tree.
%   tree: Vector of information about the tree.
% Returns:
%   preds: Vector of predictions.

num_examples = size(leaf_probs, 1);
num_classes = numel(tree);
all_probs = get_all_probs(leaf_probs, tree);
% Simply get the expected reward of each node for each image and take the max.
expected_rewards = bsxfun(@times, all_probs, rewards(:)');
[~, preds] = max(expected_rewards, [], 2);
