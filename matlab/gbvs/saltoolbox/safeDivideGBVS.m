function result = safeDivideGBVS(arg1,arg2)
% safeDivide - divides two arrays, checking for 0/0.
%
% result = safeDivide(arg1,arg2)
%    returns arg1./arg2, where 0/0 is assumed to be 0 instead of NaN.

% This file is part of the SaliencyToolbox - Copyright (C) 2006
% by Dirk Walther and the California Institute of Technology.
% The Saliency Toolbox is released under the GNU General Public 
% License. See the enclosed COPYRIGHT document for details. 
% For more information about this project see: 
% http://www.saliencytoolbox.net

ze = (arg2 == 0);
arg2(ze) = 1;
result = arg1./arg2;
result(ze) = 0;
