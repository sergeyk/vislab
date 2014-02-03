function imgShift = shiftImage( img , theta )

Worig = size(img,2);
Horig = size(img,1);

pad = 2;
imgpad = padImage(img,pad);

W = size(imgpad,2);
H = size(imgpad,1);
xi = repmat( [ 1 : W ] , [ H 1 ] );
yi = repmat( [ 1 : H ]', [ 1 W ] );

dx = cos(theta * pi / 180 );
dy =  -sin(theta * pi / 180);
imgpadshift = interp2( xi , yi , imgpad , xi + dx , yi + dy );

imgShift = imgpadshift( pad + 1 : pad + Horig , pad + 1 : pad + Worig );