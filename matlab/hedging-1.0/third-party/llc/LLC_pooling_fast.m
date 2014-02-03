% ========================================================================
% Pooling the llc codes to form the image feature
% USAGE: [beta] = LLC_pooling(feaSet, B, pyramid, knn)
% Inputs
%       feaSet      -the coordinated local descriptors
%       B           -the codebook for llc coding
%       pyramid     -the spatial pyramid structure
%       knn         -the number of neighbors for llc coding
% Outputs
%       beta        -the output image feature
%
% Written by Jianchao Yang @ IFP UIUC
% May, 2010
% ========================================================================

function [beta] = LLC_pooling_fast(feaSet, B, pyramid, knn, flann_options)


if ~exist('flann_options', 'var') || isempty(flann_options)
    flann_options=[];
end

dSize = size(B, 2);
nSmp = size(feaSet.feaArr, 2);

img_width = feaSet.width;
img_height = feaSet.height;
idxBin = zeros(nSmp, 1);

% llc coding
max_mat_size=20*1024*1024;

if ~isempty(flann_options)
    max_mat_size=200*1024*1024;
end


block_size = ceil(max_mat_size / dSize);
num_block = ceil(nSmp / block_size);

if ~isempty(flann_options)
    block_size = nSmp;
    num_block=1;
end


llc_codes = sparse(dSize, nSmp);
for i = 1:num_block
    block_start = (i-1) * block_size + 1;
    block_end = min(nSmp, block_start + block_size -1 );
    llc_codes(:,block_start:block_end) = sparse(LLC_coding_appr_fast(B', feaSet.feaArr(:,block_start:block_end)', knn, [], flann_options)');
end
    %llc_codes = llc_codes';

% spatial levels
pLevels = length(pyramid);
% spatial bins on each level
pBins = pyramid.^2;
% total spatial bins
tBins = sum(pBins);

beta = zeros(dSize, tBins);
bId = 0;

for iter1 = 1:pLevels,
    
    nBins = pBins(iter1);
    
    wUnit = img_width / pyramid(iter1);
    hUnit = img_height / pyramid(iter1);
    
    % find to which spatial bin each local descriptor belongs
    xBin = ceil(feaSet.x / wUnit);
    yBin = ceil(feaSet.y / hUnit);
    idxBin = (yBin - 1)*pyramid(iter1) + xBin;
    
    for iter2 = 1:nBins,     
        bId = bId + 1;
        sidxBin = find(idxBin == iter2);
        if isempty(sidxBin),
            continue;
        end      
        beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);
    end
end

if bId ~= tBins,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
