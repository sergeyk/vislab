
% this function creates the weight matrix for making edge weights
% and saves some other constants (like node-in-lattice index) to a 'frame'
% used when the graphs are made from saliency/feature maps.

%
% edge types (by default, instantiate fully connected graph. 
%  use inter/intra-type = 1 to connect only to nearest neighbors)
%
% inter_type :
%   1 => only same-location neighbor
%   2 => everywhere inter-scale on consecutive scales
% intra_type :
%   1 => only neighbor
%   2 => everywhere
% cyclic_type :
%   1 => cyclic boundary rules
%   2 => non-cyclic boundaries
%
%  jharel 7 / 27 / 06

function [frame] = graphsalinit( map_size , multilevels , inter_type , intra_type , cyclic_type )

dims = getDims( map_size , multilevels );
[N,nam] = namenodes(dims);
lx = makeLocationMap( dims , nam , N );
cx = connectMatrix( dims , lx , inter_type , intra_type , cyclic_type );
dx = distanceMatrix( dims , lx , cyclic_type );
D = cx .* dx;

frame.D = D;
frame.lx = lx;
frame.dims = dims;
frame.multilevels = multilevels;
