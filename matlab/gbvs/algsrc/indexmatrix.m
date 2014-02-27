function ix = indexmatrix( dims )

ix = zeros(dims);
p = prod(dims);

for i=1:p
 ix(i) = i;
end
