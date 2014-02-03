%
% gives you a matrix where
% d( ix(i,j) , ix(ii,jj) ) = distance^2 between (i,j) & (ii,jj)
%
% cyclic_type
%   1 => cyclic boundary rules
%   2 => non-cyclic boundaries

function d = simpledistance( dim , cyclic_type )

d = 0;

ix = indexmatrix( dim );
N = prod( dim );

d = zeros( N , N );

for i=1:dim(1)
    for j=1:dim(2)
        for ii=1:dim(1)
            for jj=1:dim(2)
                if ( d( ix(i,j) , ix(ii,jj) ) == 0 )
                    di = 0 ; dj = 0;
                    if ( cyclic_type==1 )
                        di = min( abs(i-ii) , abs( abs(i-ii) - dim(1) ) );
                        dj = min( abs(j-jj) , abs( abs(j-jj) - dim(2) ) );
                    else
                        di = i-ii;
                        dj = j-jj;
                    end
                    d( ix(i,j) , ix(ii,jj) ) = di^2 + dj^2;
                    d( ix(ii,jj) , ix(i,j) ) = di^2 + dj^2;
                end
            end
        end
    end
end

