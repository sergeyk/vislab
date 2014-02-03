function out = M_motion( param, img , img_prev, prev_img_shift , ti )

if ( nargin == 1 )
    out.weight = param.motionWeight;
    out.numtypes = length( param.motionAngles );
    for i = 1 : length( param.motionAngles ),
        out.descriptions{i} = sprintf('Motion Direction %g',param.motionAngles(i));
    end    
else    
    out.imgShift = shiftImage( img , param.motionAngles(ti) );    
    out.map = abs( img .* prev_img_shift - img_prev .* out.imgShift );
    
    % this rule comes from:
    % http://ilab.usc.edu/publications/doc/Itti_etal03spienn.pdf
    % "Values smaller than 3.0 are set to zero"
    % note: this doens't seem to work ? (9/4/09)
    % out.map( out.map < 3/255 ) = 0;    
end
