function out = C_color( fparam, img , imgR, imgG, imgB, typeidx )

if ( nargin == 1 )

  out.weight = fparam.colorWeight;
  
  out.numtypes = 2;
  out.descriptions{1} = 'Blue-Yellow';
  out.descriptions{2} = 'Red-Green';

else
  if ( typeidx ) == 1
    out.map = safeDivideGBVS( abs(imgB-min(imgR,imgG)) , img );
  else
    out.map = safeDivideGBVS( abs(imgR-imgG) , img );
  end
end
