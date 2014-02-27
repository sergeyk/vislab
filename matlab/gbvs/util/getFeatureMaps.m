function [rawfeatmaps, motionInfo] = getFeatureMaps( img , param , prevMotionInfo )

%
% this computes feature maps for each cannnel in featureChannels/
%

load mypath;

%%%%
%%%% STEP 1 : form image pyramid and prune levels if pyramid levels get too small.
%%%%

mymessage(param,'forming image pyramid\n');

levels = [ 2 : param.maxcomputelevel ];

is_color = (size(img,3) == 3);
imgr = []; imgg = []; imgb = [];
if ( is_color ) [imgr,imgg,imgb,imgi] = mygetrgb( img );
else imgi = img; end

imgL = {};
imgL{1} = mySubsample(imgi);
imgR{1} = mySubsample(imgr); imgG{1} = mySubsample(imgg); imgB{1} = mySubsample(imgb);

for i=levels

    imgL{i} = mySubsample( imgL{i-1} );
    if ( is_color )
        imgR{i} = mySubsample( imgR{i-1} );
        imgG{i} = mySubsample( imgG{i-1} );
        imgB{i} = mySubsample( imgB{i-1} );
    else
        imgR{i} = []; imgG{i} = []; imgB{i} = [];
    end
    if ( (size(imgL{i},1) < 3) | (size(imgL{i},2) < 3 ) )
        mymessage(param,'reached minimum size at level = %d. cutting off additional levels\n', i);
        levels = [ 2 : i ];
        param.maxcomputelevel = i;
        break;
    end

end

%%% update previous frame estimate based on new frame
if ( (param.flickerNewFrameWt == 1) || (isempty(prevMotionInfo) ) )
    motionInfo.imgL = imgL;
else    
    w = param.flickerNewFrameWt;    
    for i = levels,
        %%% new frame gets weight flickerNewFrameWt
        motionInfo.imgL =  w * imgL{i} + ( 1 - w ) * prevMotionInfo.imgL{i};
    end
end
    
%%%
%%% STEP 2 : compute feature maps
%%%

mymessage(param,'computing feature maps...\n');

rawfeatmaps = {};

%%% get channel functions in featureChannels/directory

channel_files = dir( [pathroot '/util/featureChannels/*.m'] );

motionInfo.imgShifts = {};

for ci = 1 : length(channel_files)
  
    %%% parse the channel letter and name from filename
    parts = regexp( channel_files(ci).name , '^(?<letter>\w)_(?<rest>.*?)\.m$' , 'names');
    if ( isempty(parts) ), continue; end % invalid channel file name
    
    channelLetter = parts.letter;
    channelName = parts.rest;
    channelfunc = str2func(sprintf('%s_%s',channelLetter,channelName));
    useChannel = sum(param.channels==channelLetter) > 0;

    if ( ((channelLetter == 'C') || (channelLetter=='D')) && useChannel && (~is_color) )
        mymessage(param,'oops! cannot compute color channel on black and white image. skipping this channel\n');
        continue;
    elseif (useChannel)

        mymessage(param,'computing feature maps of type "%s" ... \n', channelName);

        obj = {};
        obj.info = channelfunc(param);
        obj.description = channelName;

        obj.maps = {};
        obj.maps.val = {};

        %%% call the channelfunc() for each desired image resolution (level in pyramid)
        %%%  and for each type index for this channel.

        for ti = 1 : obj.info.numtypes            
            obj.maps.val{ti} = {};
            mymessage(param,'..pyramid levels: ');
            for lev = levels,                
                mymessage(param,'%d (%d x %d)', lev, size(imgL{lev},1), size(imgL{lev},2));                
                if ( (channelLetter == 'F') || (channelLetter == 'M') )                   
                    if ( ~isempty(prevMotionInfo) )
                        prev_img = prevMotionInfo.imgL{lev};
                    else
                        prev_img = imgL{lev};
                    end
                    
                    if ( ~isempty(prevMotionInfo) && isfield(prevMotionInfo,'imgShifts') && (channelLetter == 'M') )
                      prev_img_shift = prevMotionInfo.imgShifts{ti}{lev};
                    else
                      prev_img_shift = 0;
                    end

                    map = channelfunc(param,imgL{lev},prev_img,prev_img_shift,ti);                    
                    if (isfield(map,'imgShift'))
                       motionInfo.imgShifts{ti}{lev} = map.imgShift; 
                    end                    
                else
                    map = channelfunc(param,imgL{lev},imgR{lev},imgG{lev},imgB{lev},ti);
                end    
                obj.maps.origval{ti}{lev} = map.map;
                map = imresize( map.map , param.salmapsize , 'bicubic' );
                obj.maps.val{ti}{lev} = map;
            end
            mymessage(param,'\n');
        end

        %%% save output to rawfeatmaps structure
        eval( sprintf('rawfeatmaps.%s = obj;', channelName) );

    end
end

