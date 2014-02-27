function [out,motionInfo] = gbvs(img,param,prevMotionInfo)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                                                     %                            
% This computes the GBVS map for an image and puts it in master_map.                                  %
%                                                                                                     %
% If this image is part of a video sequence, motionInfo needs to be recycled in a                     %
% loop, and information from the previous frame/image will be used if                                 %
% "flicker" or "motion" channels are employed.                                                        %
% You need to initialize prevMotionInfo to [] for the first frame  (see demo/flicker_motion_demo.m)   %
%                                                                                                     %
%  input                                                                                              %
%    - img can be a filename, or image array (double or uint8, grayscale or rgb)                      %
%    - (optional) param contains parameters for the algorithm (see makeGBVSParams.m)                  %
%                                                                                                     %
%  output structure 'out'. fields:                                                                    %
%    - master_map is the GBVS map for img. (.._resized is the same size as img)                       %
%    - feat_maps contains the final individual feature maps, normalized                               %
%    - map_types contains a string description of each map in feat_map (resp. for each index)         %
%    - intermed_maps contains all the intermediate maps computed along the way (act. & norm.)         %
%      which are used to compute feat_maps, which is then combined into master_map                    %
%    - rawfeatmaps contains all the feature maps computed at the various scales                       %
%                                                                                                     %
%  Jonathan Harel, Last Revised Aug 2008. jonharel@gmail.com                                          %
%                                                                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ( strcmp(class(img),'char') == 1 ) img = imread(img); end
if ( strcmp(class(img),'uint8') == 1 ) img = double(img)/255; end
if ( (size(img,1) < 128) || (size(img,2) < 128) )
    fprintf(2,'GBVS Error: gbvs() meant to be used with images >= 128x128\n');
    out = [];
    return;
end

if ( (nargin == 1) || (~exist('param')) || isempty(param) ) param = makeGBVSParams; end
[grframe,param] = initGBVS(param,size(img));


if ( (nargin < 3) || (~exist('prevMotionInfo')) )
    prevMotionInfo = [];
end

if ( param.useIttiKochInsteadOfGBVS )
    mymessage(param,'NOTE: Computing STANDARD Itti/Koch instead of Graph-Based Visual Saliency (GBVS)\n\n');
end

%%%% 
%%%% STEP 1 : compute raw feature maps from image
%%%%

mymessage(param,'computing feature maps...\n');
if ( size(img,3) == 3 ) imgcolortype = 1; else, imgcolortype = 2; end
[rawfeatmaps motionInfo] = getFeatureMaps( img , param , prevMotionInfo );

%%%% 
%%%% STEP 2 : compute activation maps from feature maps
%%%%

mapnames = fieldnames(rawfeatmaps);
mapweights = zeros(1,length(mapnames));
map_types = {};
allmaps = {};
i = 0;
mymessage(param,'computing activation maps...\n');
for fmapi=1:length(mapnames)
    mapsobj = eval( [ 'rawfeatmaps.' mapnames{fmapi} ';'] );
    numtypes = mapsobj.info.numtypes;
    mapweights(fmapi) = mapsobj.info.weight;
    map_types{fmapi} = mapsobj.description;
    for typei = 1 : numtypes
        if ( param.activationType == 1 )
            for lev = param.levels                
                mymessage(param,'making a graph-based activation (%s) feature map.\n',mapnames{fmapi});
                i = i + 1;
                [allmaps{i}.map,tmp] = graphsalapply( mapsobj.maps.val{typei}{lev} , ...
                    grframe, param.sigma_frac_act , 1 , 2 , param.tol );
                allmaps{i}.maptype = [ fmapi typei lev ];
            end
        else
            for centerLevel = param.ittiCenterLevels
                for deltaLevel = param.ittiDeltaLevels
                    mymessage(param,'making a itti-style activation (%s) feature map using center-surround subtraction.\n',mapnames{fmapi});
                    i = i + 1;                    
                    center_ = mapsobj.maps.origval{typei}{centerLevel};
                    sz_ = size(center_);
                    surround_ = imresize( mapsobj.maps.origval{typei}{centerLevel+deltaLevel}, sz_ , 'bicubic' );                    
                    allmaps{i}.map = (center_ - surround_).^2;
                    allmaps{i}.maptype = [ fmapi centerLevel deltaLevel ];
                end
            end
        end
    end
end

    
%%%% 
%%%% STEP 3 : normalize activation maps
%%%%

mymessage(param,'normalizing activation maps...\n');
norm_maps = {};
for i=1:length(allmaps)
    mymessage(param,'normalizing a feature map (%d)... ', i);
    if ( param.normalizationType == 1 )
        mymessage(param,' using fast raise to power scheme\n ', i);
        algtype = 4;
        [norm_maps{i}.map,tmp] = graphsalapply( allmaps{i}.map , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );        
    elseif ( param.normalizationType == 2 )
        mymessage(param,' using graph-based scheme\n');
        algtype = 1;
        [norm_maps{i}.map,tmp] = graphsalapply( allmaps{i}.map , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );                
    else
        mymessage(param,' using global - mean local maxima scheme.\n');
        norm_maps{i}.map = maxNormalizeStdGBVS( mat2gray(imresize(allmaps{i}.map,param.salmapsize, 'bicubic')) );
    end
    norm_maps{i}.maptype = allmaps{i}.maptype;
end

%%%% 
%%%% STEP 4 : average across maps within each feature channel
%%%%

comb_norm_maps = {};
cmaps = {};
for i=1:length(mapnames), cmaps{i}=0; end
Nfmap = cmaps;

mymessage(param,'summing across maps within each feature channel.\n');
for j=1:length(norm_maps)
  map = norm_maps{j}.map;
  fmapi = norm_maps{j}.maptype(1);
  Nfmap{fmapi} = Nfmap{fmapi} + 1;
  cmaps{fmapi} = cmaps{fmapi} + map;
end
%%% divide each feature channel by number of maps in that channel
for fmapi = 1 : length(mapnames)
  if ( param.normalizeTopChannelMaps) 
      mymessage(param,'Performing additional top-level feature map normalization.\n');
      if ( param.normalizationType == 1 )
          algtype = 4;
          [cmaps{fmapi},tmp] = graphsalapply( cmaps{fmapi} , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );
      elseif ( param.normalizationType == 2 )
          algtype = 1;
          [cmaps{fmapi},tmp] = graphsalapply( cmaps{fmapi} , grframe, param.sigma_frac_norm, param.num_norm_iters, algtype , param.tol );
      else
        cmaps{fmapi} = maxNormalizeStdGBVS( cmaps{fmapi} );
      end
  end
  comb_norm_maps{fmapi} = cmaps{fmapi};
end

%%%% 
%%%% STEP 5 : sum across feature channels
%%%%

mymessage(param,'summing across feature channels into master saliency map.\n');
master_idx = length(mapnames) + 1;
comb_norm_maps{master_idx} = 0;
for fmapi = 1 : length(mapnames)
  mymessage(param,'adding in %s map with weight %0.3g (max = %0.3g)\n', map_types{fmapi}, mapweights(fmapi) , max( cmaps{fmapi}(:) ) );
  comb_norm_maps{master_idx} = comb_norm_maps{master_idx} + cmaps{fmapi} * mapweights(fmapi);
end
master_map = comb_norm_maps{master_idx};
master_map = attenuateBordersGBVS(master_map,4);
master_map = mat2gray(master_map);

%%%%
%%%% STEP 6: blur for better results
%%%%
blurfrac = param.blurfrac;
if ( param.useIttiKochInsteadOfGBVS )
  blurfrac = param.ittiblurfrac;
end
if ( blurfrac > 0 )
  mymessage(param,'applying final blur with with = %0.3g\n', blurfrac);
  k = mygausskernel( max(size(master_map)) * blurfrac , 2 );
  master_map = myconv2(myconv2( master_map , k ),k');
  master_map = mat2gray(master_map);
end

if ( param.unCenterBias )  
  invCB = load('invCenterBias');
  invCB = invCB.invCenterBias;
  centerNewWeight = 0.5;
  invCB = centerNewWeight + (1-centerNewWeight) * invCB;
  invCB = imresize( invCB , size( master_map ) );
  master_map = master_map .* invCB;
  master_map = mat2gray(master_map);
end

%%%% 
%%%% save descriptive, rescaled (0-255) output for user
%%%%

feat_maps = {};
for i = 1 : length(mapnames)
  feat_maps{i} = mat2gray(comb_norm_maps{i});
end

intermed_maps = {};
for i = 1 : length(allmaps)
 allmaps{i}.map = mat2gray( allmaps{i}.map );
 norm_maps{i}.map = mat2gray( norm_maps{i}.map );
end

intermed_maps.featureActivationMaps = allmaps;
intermed_maps.normalizedActivationMaps = norm_maps;
master_map_resized = mat2gray(imresize(master_map,[size(img,1) size(img,2)]));

out = {};
out.master_map = master_map;
out.master_map_resized = master_map_resized;
out.top_level_feat_maps = feat_maps;
out.map_types = map_types;
out.intermed_maps = intermed_maps;
out.rawfeatmaps = rawfeatmaps;
out.paramsUsed = param;
if ( param.saveInputImage )
    out.inputimg = img;
end
