function [gt_map, ids, gt_weight] = read_gt(filename)

gt_map = java.util.Hashtable;

fd = fopen(filename);

c = textscan(fd,'%s %s');

fclose(fd);

n = numel(c{1});

ids = cell(n,1);

gt_weight = java.util.Hashtable;

for i =1:n
    gt_map.put(c{1}{i}, c{2}{i});
    ids{i} = c{1}{i};
end

for i = 1:n
    w = gt_weight.get(ids{i});
    if isempty(w)
        gt_weight.put(ids{i}, 1);
    else
        gt_weight.put(ids{i}, w+1);
    end
end



