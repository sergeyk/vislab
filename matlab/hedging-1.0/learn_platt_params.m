function [platt_a, platt_b] = learn_platt_params(decision_values, labels, tree)
% [platt_a, platt_b] = learn_platt_params(decision_values, labels, tree):
% Learns Platt scaling parameters.
% Platt scaling is a way to convert decision values into probabilities by
% fitting a sigmoid function to the decision values for each class.
% It does so by optimizing for 'a' and 'b' in the following expression:
%    1 / (1 + exp(a * z + b))
% where 'z' is an input decision value.
% Arguments:
%   decision_values: num_images x num_classifiers matrix of decision values.
%     decision_values(i, j) gives the decision value for classifier j on
%     image i.  The classifiers should be in order.
%   labels: num_images x 1 vector of labels.  It is assumed that the labels
%     occupy the range 1:num_classes and each label occurs at least once.
%   tree: Vector of metadata about the tree.
% Returns:
%   platt_a: num_classifiers x 1 vector of parameters 'a' in Platt scaling,
%     ordered by classifier number.
%   platt_b: num_classifiers x 1 vector of parameters 'b' in Platt scaling,
%     ordered by classifier number.

num_classes = size(decision_values, 2);
gt_vec = labels_to_vec(labels, num_classes);
gt_vec_full = get_all_probs(gt_vec, tree);
platt_a = zeros(num_classes, 1);
platt_b = zeros(num_classes, 1);
for i = 1:num_classes
  % Convert [0,1] to [-1,1]
  class_labels = gt_vec_full(:, i) * 2 - 1;
  class_decision_values = decision_values(:, i);
  [class_a, class_b] = plattscaling(class_decision_values, ...
    class_labels);
  platt_a(i) = class_a;
  platt_b(i) = class_b;
end
