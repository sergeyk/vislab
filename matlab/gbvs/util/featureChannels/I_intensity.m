function out = I_intensity( fparam , img , imgR, imgG, imgB, typeidx )

if ( nargin == 1)  
  out.weight = fparam.intensityWeight;
  out.numtypes = 1;
  out.descriptions{1} = 'Intensity';    
else
  out.map = img;
end
