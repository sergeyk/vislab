function out = D_dklcolor( fparam, img , imgR, imgG, imgB, typeidx )

% CHROMATIC MECHANISMS IN LATERAL 
% GENICULATE NUCLEUS OF MACAQUE 
% BY A. M. DERRINGTON, J. KRAUSKOPF AND P. LENNIE 

% from the abstract:
%  (a) an axis along 
%  which only luminance varies, without change in chromaticity, (b) a 'constant B' axis 
%  along which chromaticity varies without changing the excitation of blue-sensitive (B) 
%  cones, (c) a 'constant R & G' axis along which chromaticity varies without change 
%  in the excitation of red-sensitive (R) or green-sensitive (G) cones

if ( nargin == 1 )
  out.weight = fparam.dklcolorWeight;
  out.numtypes = 3;
  out.descriptions{1} = 'DKL Luminosity Channel';
  out.descriptions{2} = 'DKL Color Channel 1';
  out.descriptions{3} = 'DKL Color Channel 2';
else
  rgb = repmat( imgR , [ 1 1 3 ] );
  rgb(:,:,2) = imgG;
  rgb(:,:,3) = imgB;
  dkl = rgb2dkl( rgb );

  if ( typeidx == 1 )
    out.map = dkl(:,:,1);
  elseif ( typeidx == 2 )
    out.map = dkl(:,:,2);
  elseif ( typeidx == 3 )
    out.map = dkl(:,:,3);
  end
end
