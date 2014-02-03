function [A,tmp] = areaROC( p )
tmp = -1;
p = getBestRows(p);
xy = sortrows([p(:,2) p(:,1)]);

x = xy(:,1);
y = xy(:,2);

x = [ 0 ; x ; 1 ];
y = [ 0 ; y ; 1 ];

A = trapz( x , y );

