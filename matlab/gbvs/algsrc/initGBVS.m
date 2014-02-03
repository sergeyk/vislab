%
% some constants used across different calls to gbvs()
%

function [grframe,param] = initGBVS(param, imgsize)

mymessage(param,'initializing....\n');

% logical consistency checking of parameters
if ( min(param.levels) < 2 )
    mymessage(param,'oops. cannot use level 1.. trimming levels used\n');
    param.levels = param.levels(param.levels>1);
end
if ( param.useIttiKochInsteadOfGBVS )
    param.activationType = 2;
    param.normalizationType = 3;
    param.normalizeTopChannelMaps = 1;
end

param.maxcomputelevel = max(param.levels);
if (param.activationType==2)
    param.maxcomputelevel = max( param.maxcomputelevel , max(param.ittiCenterLevels)+max(param.ittiDeltaLevels) );
end

w = imgsize(2); h = imgsize(1); scale = param.salmapmaxsize / max(w,h);
salmapsize = round( [ h w ] * scale );

% weight matrix
if ( ~param.useIttiKochInsteadOfGBVS )
  load mypath;
  ufile = sprintf('%s__m%s__%s.mat',num2str(salmapsize),num2str(param.multilevels),num2str(param.cyclic_type));
  ufile(ufile==' ') = '_';
  ufile = fullfile( pathroot , 'initcache' ,  ufile );
  if ( exist(ufile) )
    grframe = load(ufile);
    grframe = grframe.grframe;
  else
    grframe = graphsalinit( salmapsize , param.multilevels , 2, 2, param.cyclic_type );
    save(ufile,'grframe');
  end
else
  grframe = [];
end

% gabor filters
gaborParams.stddev = 2;gaborParams.elongation = 2;
gaborParams.filterSize = -1;gaborParams.filterPeriod = pi;
for i = 1 : length(param.gaborangles)
    theta = param.gaborangles(i);
    gaborFilters{i}.g0 = makeGaborFilterGBVS(gaborParams, theta, 0);
    gaborFilters{i}.g90 = makeGaborFilterGBVS(gaborParams, theta, 90);
end

param.gaborParams = gaborParams;
param.gaborFilters = gaborFilters;
param.salmapsize = salmapsize;
param.origimgsize = imgsize;
