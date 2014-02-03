function [all_probs] = get_all_probs(leaf_probs, synsets)
%leaf probs: n * m, n examples and m synsets
%synsets is a meta structrue
%it's assumed that leaves are ordered from 1 to # of leaves.

% first find the root
hs = [synsets.height];
root = find(hs==max(hs));

n = size(leaf_probs, 1);
m = numel(synsets);
all_probs = zeros(n, m);
all_probs(:, 1:size(leaf_probs, 2)) = leaf_probs;

all_probs = get_prob(root, all_probs, synsets);

function [all_probs] = get_prob(r, all_probs, synsets)
    rnode=synsets(r);
    if rnode.num_children > 0
        p = 0;
        for i =  1:rnode.num_children
            c = rnode.children(i);
            all_probs = get_prob(c, all_probs, synsets);
            p = p + all_probs(:, c);
        end
        all_probs(:, r) = p;        
    end
