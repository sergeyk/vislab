function out = F_flicker( param, img , img_prev, prev_img_shift , ti )

if ( nargin == 1 )
    out.weight = param.flickerWeight;
    out.numtypes = 1;
    out.descriptions{1} = 'Flicker';
else
    out.map = abs( img - img_prev );
end
