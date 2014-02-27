function dist_hist_plot(height_portions, accuracy_guarantees)
% Arguments:
%   height_portions: A num_heights * num_guarantees matrix
%   accuracy_guarantees: Vector of guarantees, used for labeling
% Returns:
%   Nothing, but makes a histograph of node height vs the portion of
%   predictions made at that each height.

% Get data for lines
num_heights = size(height_portions, 1);
assert(size(height_portions, 2) == numel(accuracy_guarantees));
bar_handle = bar(0:num_heights - 1, height_portions);

% Get some nice colors
facecolors = {[15, 127, 247] / 256, [240, 63, 43] / 256, 'y', ...
  'm', 'c', 'r', 'g', 'b', 'w', 'k'};

for i = 1:numel(accuracy_guarantees)
  set(bar_handle(i), 'facecolor', facecolors{i});
end

% Extract accuracy guarantees for the legend
labels = cell(1, numel(accuracy_guarantees));
for i = 1:numel(accuracy_guarantees)
  labels{i} = num2str(accuracy_guarantees(i));
end
h_legend = legend(labels{:}, 'Location', 'NorthEast');
xlabel('Node Height', 'fontsize', 36);
ylabel('% Predictions', 'fontsize', 36);
