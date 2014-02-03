% Plotting
load('inf_results.mat');

figure;
acc_vs_reward_plot(results, num_leaves, struct('legend_names', {methods}));

darts_results = results(1);
plot_guarantees = [.9, .99];
hist_indices = find(ismember(accuracy_guarantees, plot_guarantees));
darts_results.height_portions = darts_results.height_portions(:, hist_indices);
accuracy_guarantees = plot_guarantees;
figure;
dist_hist_plot(darts_results.height_portions, accuracy_guarantees);
