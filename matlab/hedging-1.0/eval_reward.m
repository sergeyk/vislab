function [reward, acc, height_portion, height_acc] = eval_reward(preds, ...
  labels, rewards, tree)
% eval_reward(preds, labels, rewards, tree)
% Evaluates the reward of the given predictions.
% Arguments:
%   preds: Ids of each prediction.
%   labels: Ground truth labels of each image.
%   rewards: The rewards for each node in the tree.
%   tree: Vector of information about the tree.
% Returns:
%   reward: The average reward obtained.
%   acc: The overall accuracy.
%   height_portion: The portion of all predictions at each height, in
%     order of increasing height (i.e. starting at the leaves).
%   height_accuracy: The accuracy of the predictions at each height, in
%     order of increasing height.

num_examples = numel(preds);
num_classes = numel(tree);
heights = [tree.height] + 1;
num_leaves = nnz(heights == 1);

% Compute the 'sufficient statistics', as it were, of the numbers we're going
% to return.  Vectorize the whole thing to make it fast.
pred_vec = labels_to_vec(preds, num_classes);

gt_vec = labels_to_vec(labels, num_leaves);
gt_vec_full = get_all_probs(gt_vec, tree);

correct_vec = gt_vec_full .* pred_vec;
num_right = sum(correct_vec(:));

reward_vec = bsxfun(@times, correct_vec, rewards(:)');
reward = sum(reward_vec(:));

height_pred_vec = bsxfun(@times, pred_vec, heights(:)');
height_counts = histc(height_pred_vec(:), 1:max(heights));
height_correct_vec = bsxfun(@times, correct_vec, heights(:)');
height_goods = histc(height_correct_vec(:), 1:max(heights));

reward = reward / num_examples;
acc = num_right / num_examples;
height_acc = height_goods ./ height_counts;
height_portion = height_counts ./ num_examples;
