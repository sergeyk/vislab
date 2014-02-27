

function [Anorm,iters] = graphsalapply( A , frame , sigma_frac, num_iters , algtype , tol )

%
%  this function is the heart of GBVS.
%   * it takes a feature map, forms a graph over its locations, which is either a lattice of a hierachy ("multiresolution")of lattices,
%     connects the nodes with weighted edges, and computes the equilibrium distribution over states.
%   * the weight assignment rule from node i to node j depends on the 'algtype' 
  
%  algtype    algorith type:
%               1 : MM( i->j ) = w*AL(j)               [ mass conc ]
%               2 : MM( i->j ) = w*|AL(i)-AL(j)|       [ sal diff ]
%               3 : MM( i->j ) = w*|log(AL(i)/AL(j))|  [ sal log ]
%               4 : Anorm = A . ^ w                    [ simple mass concentration ]

%  tol controls a stopping rule on the computation of the equilibrium distribution (principal eigenvector)
%  the lower it is, the longer the algorithm runs.

if ( algtype == 4 )
  Anorm = A .^ 1.5;
  iters = 1;
  return;
end
  
% form a multiresolution pyramid of feature maps according to multilevels
lx = frame.lx; 
[ Apyr , dims ] = formMapPyramid( A , frame.multilevels );

% get a weight matrix between nodes based on distance matrix
sig = sigma_frac * mean(size(A));
Dw = exp( -1 * frame.D / (2 * sig^2) );

% assign a linear index to each node
AL = mexArrangeLinear( Apyr , dims );

% create the state transition matrix between nodes
P = size(lx,1);
MM = zeros( P , P );

iters = 0;

for i=1:num_iters

  % assign edge weights based on distances between nodes and algtype  
  mexAssignWeights( AL , Dw , MM , algtype );

  % make it a markov matrix (so each column sums to 1)
  mexColumnNormalize( MM );

  % find the principal eigenvector
  [AL,iteri] = principalEigenvectorRaw( MM , tol );
  iters = iters + iteri;

end

% collapse multiresolution representation back onto one scale
Vo = mexSumOverScales( AL , lx , prod(size(A)) );

% arrange the nodes back into a rectangular map
Anorm = reshape(Vo,size(A));
