function labels = ids_to_labels(ids, gt_map, class_map)

n = numel(ids);
labels = ones(n,1);

for i = 1:n
    g = gt_map.get(ids{i});
    labels(i) = class_map.get(g);
end
        
