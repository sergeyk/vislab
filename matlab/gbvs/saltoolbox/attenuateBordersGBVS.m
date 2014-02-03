function result = attenuateBordersGBVS(data,borderSize)
% attentuateBorders - linearly attentuates the border of data.
%
% result = attenuateBorders(data,borderSize)
%   linearly attenuates a border region of borderSize
%   on all sides of the 2d data array

% This file is part of the SaliencyToolbox - Copyright (C) 2006
% by Dirk Walther and the California Institute of Technology.
% The Saliency Toolbox is released under the GNU General Public 
% License. See the enclosed COPYRIGHT document for details. 
% For more information about this project see: 
% http://www.saliencytoolbox.net

result = data;
dsz = size(data);

if (borderSize * 2 > dsz(1)) borderSize = floor(dsz(1) / 2); end
if (borderSize * 2 > dsz(2)) borderSize = floor(dsz(2) / 2); end
if (borderSize < 1) return; end

bs = [1:borderSize];
coeffs = bs / (borderSize + 1);

% top and bottom
rec = repmat(coeffs',1,dsz(2));
result(bs,:) = result(bs,:) .* rec;
range = dsz(1) - bs + 1;
result(range,:) = result(range,:) .* rec;

% left and right
rec = repmat(coeffs,dsz(1),1);
result(:,bs) = result(:,bs) .* rec;
range = dsz(2) - bs + 1;
result(:,range) = result(:,range) .* rec;
