
function a = rocScoreSaliencyVsFixations( salmap , X , Y , origimgsize )  

%
% outputs ROC Area-Under-Curve Score between a saliency map and fixations.
%
%  salmap       : a saliency map
%  X            : vector of X locations of fixations in original image coordinates
%  Y            : vector of Y locations of fixations in original image coordinates
%  origimgsize  : size of original image (should have same aspect ratio as saliency map)
%

a = rocSal( salmap , makeFixationMask( X , Y , origimgsize , size(salmap) ) );
