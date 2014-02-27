function [N,nam] = namenodes(dims)

Nmaps = size( dims , 1 );

N = 0;
for i=1:Nmaps
    N = N + prod( dims(i,:) );
end
nam = -1 * ones( [dims(1,:)  Nmaps] );
curN = 0;

for i=1:Nmaps
    for k=1:dims(i,2)
        for j=1:dims(i,1)
            nam(j,k,i) = curN;
            curN = curN + 1;
        end
    end
end
