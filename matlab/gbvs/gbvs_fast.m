%%
%% Use this instead of gbvs() if you want slightly less predictive maps
%% computed in a fraction of the time.
%%
%%

function out = gbvs_fast( img )

params = makeGBVSParams;
params.channels = 'DO';
params.gaborangles = [ 0 90 ];
params.levels = 3;
params.verbose = 0;
params.tol = 0.003;
params.salmapmaxsize = 24;
out = gbvs(img,params);
