function result = maxNormalizeStdGBVS(data)
% maxNormalizeStd - normalization based on local maxima.
% result = maxNormalizeStd(data)
%    Normalize data by multiplying it with 
%    (max(data) - avg(localMaxima))^2 as described in;
%    L. Itti, C. Koch, E. Niebur, A Model of Saliency-Based 
%    Visual Attention for Rapid Scene Analysis, IEEE PAMI, 
%    Vol. 20, No. 11, pp. 1254-1259, Nov 1998.
%
% result = maxNormalizeStd(data,minmax)
%    Specify a dynamic for the initial maximum normalization 
%    of the input data (default: [0 10]).
%
% See also maxNormalize, maxNormalizeFancy, maxNormalizeFancyFast, makeSaliencyMap.

% This file is part of the Saliency Toolbox - Copyright (C) 2005
% by Dirk Walther and the California Institute of Technology.
% The Saliency Toolbox is released under the GNU General Public 
% License. See the enclosed COPYRIGHT document for details. 
% For more information about this project see: 
% http://klab.caltech.edu/~walther/SaliencyToolbox

%
% modified by jonathan harel 2008 for GBVS code
%  .. simplified
  
M = 10;

data = mat2gray( data ) * M;
thresh = M / 10;
[lm_avg,lm_num,lm_sum] = mexLocalMaximaGBVS(data,thresh);

if (lm_num > 1)
  result = data * (M - lm_avg)^2;
elseif (lm_num == 1)
  result = data * M .^ 2;
else
  result = data;
end
