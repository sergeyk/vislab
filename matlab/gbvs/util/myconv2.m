function c = myconv2(a,b)
  
%
% conv2 with 'same' and repeating boundary condition
%

vpad = ceil(( size(b,1) - 1 ) / 2);
hpad = ceil(( size(b,2) - 1 ) / 2);
ap = padImage( a , vpad , hpad );
cp = conv2( ap , b , 'same' );
c = cp( vpad + 1 : vpad + size(a,1) , hpad + 1 : hpad + size(a,2) );

