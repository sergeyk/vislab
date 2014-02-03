function vec = labels_to_vec(labels, num_classes)
% labels_to_vec(labels, num_classes)
% Converts a list of labels into a vectorized, indicator function-like form.
% Arguments:
%   labels: Vector of class labels.
%   num_classes: The total number of classes
% Returns:
%   vec: A num_labels * num_classes matrix.  Row i of 'vec' contains a 1
%     in column labels(i) and has zeros elsewhere.

vec = zeros(numel(labels), num_classes);
vec(sub2ind(size(vec), 1:numel(labels), labels(:)')) = 1;
