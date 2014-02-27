
%
% computes the principal eigenvector of a [nm nm] markov matrix
%
% j harel 6/06

function [v,iter] = principalEigenvectorRaw( markovA , tol )

if ( sparseness(markovA) < .4 )
     markovA = sparse(markovA);
end

D = size(markovA,1);


df = 1;
v = ones(size(markovA,1),1)/D;
oldv = v;
oldoldv = v;
iter = 0;

while ( df > tol )
    oldv = v;
    oldoldv = oldv;
    v = markovA * v;
    df = norm(oldv-v);
    iter = iter + 1;
    s = sum(v);
    if ( s >= 0 &&  s < inf )
        continue;
    else
        v = oldoldv;
        break;
    end
end

v = v / sum(v);


