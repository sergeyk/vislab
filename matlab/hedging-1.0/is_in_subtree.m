function in_subtree = is_in_subtree(tree, node1, node2)
% is_in_subtree(tree, node1, node2)
% Determines whether 'node1' is a descendent of 'node2' in the tree
% described by 'tree'.
% Arguments:
%   tree: Vector of information about the tree.
%   node1: Id of the possible descendent node.
%   node2: Id of the possible ancestor node.
in_subtree = node1 == node2;
while ~isempty(tree(node1).parent)
  in_subtree = in_subtree || (tree(node1).parent == node2);
  node1 = tree(node1).parent;
end
