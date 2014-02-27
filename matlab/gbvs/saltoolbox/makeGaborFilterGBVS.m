function filter = makeGaborFilterGBVS(gaborParams, angle, phase, varargin)
% makeGaborFilter - returns a 2d Gabor filter.
%
% filter = makeGaborFilter(gaborParams, angle, phase, makeDisc)
%    Returns a two-dimensional Gabor filter with the parameter:
%    gaborParams - a struct with the following fields:
%       filterPeriod - the period of the filter in pixels,
%       elongation - the ratio of length versus width,
%       filterSize - the size of the filter in pixels,
%       stddev - the standard deviation of the Gaussian in pixels.
%    angle - the angle of orientation, in degrees,
%    phase - the phase of the filter, in degrees,
%    makeDisc - if 1, enforce a disc-shaped filter, i.e. set all values
%               outside of a circle with diameter gaborParams.filterSize to 0.
%
% filter = makeGaborFilter(gaborParams, angle, phase)
%    Returns a two-dimensional Gabor filter, assuming makeDisc = 0.
%
% See also gaborFilterMap, defaultSaliencyParams.

% This file is part of the Saliency Toolbox - Copyright (C) 2006
% by Dirk Walther and the California Institute of Technology.
% The Saliency Toolbox is released under the GNU General Public 
% License. See the enclosed COPYRIGHT document for details. 
% For more information about this project see: 
% http://www.saliencytoolbox.net

if isempty(varargin)
  makeDisc = 0;
else
  makeDisc = varargin{1};
end

% repare parameters
major_stddev = gaborParams.stddev;
minor_stddev = major_stddev * gaborParams.elongation;
max_stddev = max(major_stddev,minor_stddev);

sz = gaborParams.filterSize;
if (sz == -1)
  sz = ceil(max_stddev*sqrt(10));
else
  sz = floor(sz/2); 
end

psi = pi / 180 * phase;
rtDeg = pi / 180 * angle;

omega = 2 * pi / gaborParams.filterPeriod;
co = cos(rtDeg);
si = -sin(rtDeg);
major_sigq = 2 * major_stddev^2;
minor_sigq = 2 * minor_stddev^2;

% prepare grids for major and minor components
vec = [-sz:sz];
vlen = length(vec);
vco = vec*co;
vsi = vec*si;

major = repmat(vco',1,vlen) + repmat(vsi,vlen,1);
major2 = major.^2;
minor = repmat(vsi',1,vlen) - repmat(vco,vlen,1);
minor2 = minor.^2;

% create the actual filter
result = cos(omega * major + psi) .* ...
exp(-major2 / major_sigq ...
    -minor2 / minor_sigq);

% enforce disc shape?
if (makeDisc)
  result((major2+minor2) > (gaborParams.filterSize/2)^2) = 0;
end

% normalization
filter = result - mean(result(:));
filter = filter / sqrt(sum(filter(:).^2));
