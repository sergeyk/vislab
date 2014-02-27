function rewards = info_rewards(tree)
% Sets the rewards according to information gain, assuming a uniform
% distribution on leaf nodes.  Log2 is used.
% Arguments:
%   tree: Vector of information about each node.  It is assumed that the
%     elements of 'tree' are in ascending order, with the leaf nodes at the
%     beginning.
% Returns:
%   rewards: Column vector of rewards, indexed in the same order as 'tree'.

heights = [tree.height];
root_index = find(heights == max(heights));
num_classes = numel(tree);

% A leaf is a descendant of itself.
is_leaf = heights == 0;
num_leaf_descendants = get_all_probs(is_leaf, tree);

num_leaves = nnz(is_leaf);
rewards = zeros(num_classes, 1);
for i = 1:num_classes
  rewards(i) = log2(num_leaves / num_leaf_descendants(i));
end

