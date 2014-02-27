function out = R_contrast( fparam , img , imgR, imgG, imgB, typeidx )

if ( nargin == 1 )

  out.weight = fparam.contrastWeight;
  out.numtypes = 1;
  out.descriptions{1} = 'Intensity Contrast';  

else

  out.map = myContrast( img , round(size(img,1) * fparam.contrastwidth) );

end
