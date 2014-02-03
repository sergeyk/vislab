function k = mygausskernel( std , nstds )

maxi = round(std * nstds);
a = [ normpdf(0,0,std) zeros(1,maxi) ];

for i = 1 : maxi
    a(1+i) = normpdf( i , 0 , std );
end

k = [ a(end:-1:2) a ];
k = k / sum(k);
