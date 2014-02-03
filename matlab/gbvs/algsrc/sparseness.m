function s = sparseness(a)
s = sum(sum(a~=0)) / prod(size(a));