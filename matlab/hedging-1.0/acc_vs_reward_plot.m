function acc_vs_reward_plot(results, num_leaves, options)
% Plot achieved accuracy versus reward.
% Arguments:
%   results: Struct array of result structs.  Each result struct should contain
%     the fields 'rewards' and 'accuracies', whose format is described in
%     DARTS_eval, among other places.
%   num_leaves: Number of leaves in the semantic hierarchy.  Used to generate
%     the equivalent number of classes in the y labels.
%   options: Plotting options.
% Returns:
%   Nothing, just plots the results.

% Parse plotting options
options = add_field_if_not_present(options, 'linespecs', ...
  {'-rs', '-bs', '-gs', '-ms', '-ks', '-cs', '-ys', '-ws'});
options = add_field_if_not_present(options, 'line_width', 3);
options = add_field_if_not_present(options, 'marker_size', 10);
options = add_field_if_not_present(options, 'marker_face_colors', ...
  {'r', 'b', 'g', 'm', 'k', 'c', 'y', 'w'});
options = add_field_if_not_present(options, 'legend_names', {});
options = add_field_if_not_present(options, 'legend_location', 'SouthWest');
options = add_field_if_not_present(options, 'legend_font_size', 20);
options = add_field_if_not_present(options, 'tick_font_size', 16);
options = add_field_if_not_present(options, 'x_font_size', 36);
options = add_field_if_not_present(options, 'y_font_size', 36);
options = add_field_if_not_present(options, 'xlim', []);
options = add_field_if_not_present(options, 'ylim', []);
options = add_field_if_not_present(options, 'yint', .05);
options = add_field_if_not_present(options, 'grid', 'off');
options = add_field_if_not_present(options, 'title', '');
options = add_field_if_not_present(options, 'title_font_size', 36);

% Draw data
num_lines = numel(results);
for i = 1:num_lines
  plot([results(i).accuracies], [results(i).rewards], options.linespecs{i}, ...
    'LineWidth', options.line_width, 'MarkerSize', options.marker_size, ...
    'MarkerFaceColor', options.marker_face_colors{i});
  hold on;
end
h_legend = legend(options.legend_names{:}, 'Location', options.legend_location);

% Set font sizes
set(h_legend, 'FontSize', options.legend_font_size);
set(gca,'FontSize', options.tick_font_size);
xlhand = get(gca, 'xlabel'); %make a handle for the x axis label
xlabel('Accuracy', 'fontsize', 20);
set(xlhand, 'fontsize', options.x_font_size);
ylhand = get(gca,'ylabel');
ylabel('Info. Gain[# Classes]', 'fontsize', 20);
set(ylhand, 'fontsize', options.y_font_size)

% Axis bounds
if ~isempty(options.xlim)
  xlim(options.xlim);
end
% Get reward bounds in order to make an intelligent ylim guess
min_reward = Inf;
max_reward = -Inf;
for i = 1:num_lines
  min_reward = min([min_reward; results(i).rewards]);
  max_reward = max([max_reward; results(i).rewards]);
end
ymin = floor(1 / options.yint * min_reward) * options.yint;
ymax = ceil(1 / options.yint * max_reward) * options.yint;
if ~isempty(options.ylim)
  ylim(options.ylim);
  ymin = options.ylim(1);
  ymax = options.ylim(2);
end
% Y tick labels, including the equivalent number of classes in brackets.
y = ymin:options.yint:ymax;
set(gca, 'YTick', y);
ylabels = vertcat(y, num_leaves * 2.^(-y * log2(num_leaves)));
set(gca, 'YTickLabel', sprintf('%g[%2.1f]|', ylabels));


% Title and grid
grid(options.grid);
if ~isempty(options.title)
  title(options.title, 'fontsize', options.title_font_size);
end
end

