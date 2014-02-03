function ix = partitionindex( N , M )

binsize = floor( N / M );
leftover = N - binsize * M;

pad = zeros( 1 , M );
pad(1) = floor(leftover / 2);
pad(M) = ceil(leftover / 2);
ix = zeros( 2 , N );

curindex = 0;
for i = 1 : M
    for j = 1 : binsize + pad(i)
        curindex = curindex + 1;
        ix( : , curindex ) = [ curindex ; i ];
    end
end