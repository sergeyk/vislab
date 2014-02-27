function lx = makeLocationMap( dims , nam , N )

Nmaps = size(dims,1);
lx = zeros(N , 2^13 + 3 );

px = {};
for i = 1 : Nmaps
    px{i}.r = partitionindex( dims(1,1) , dims(i,1) );
    px{i}.c = partitionindex( dims(1,2) , dims(i,2) );
end

for i = 1 : Nmaps
    for j = 1 : dims(i,1)
        for k = 1 : dims(i,2)

            nm = nam(j,k,i);

            xcoords = px{i}.r( 1 , find( px{i}.r(2,:) == j ) );
            ycoords = px{i}.c( 1 , find( px{i}.c(2,:) == k ) );

            lst = [];
            Nx = length(xcoords);
            Ny = length(ycoords);
            Nl = 0;

            for ii=1:Nx
                for jj=1:Ny
                    lst(end+1) = nam( xcoords(ii) , ycoords(jj) , 1 );
                    Nl = Nl + 1;
                end
            end
            lx( nm + 1 , 1:Nl+2  ) = [ nm Nl lst ];
        end
    end
end

maxL = max( lx(:,2) );
lx = lx( : , 1 : maxL + 2 );
