
pathroot = pwd;
save -mat util/mypath.mat pathroot
addpath(genpath( pathroot ), '-begin');
savepath