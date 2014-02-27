function lmap = linearmap(map)
[n,m] = size(map);
lmap = reshape(map, [1 n*m]);

