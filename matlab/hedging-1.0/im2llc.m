function beta = im2llc(I,imsizes, binSizes, steps,codebook, pyramid, knn, flann_options)
% im must be float valued

    width = size(I,2);
    height = size(I,1);

    descs = [];
    frames = [];
    scales = []; % in terms of bin sizes

    for j = 1:numel(binSizes)
        im = imresize(I, imsizes(j));
        [myframes, mydescs] = vl_dsift(im,'Size',binSizes(j),'Step',steps(j),'Norm');
        frames = [frames, myframes / imsizes(j)];
        descs = [descs, mydescs];
        scales = [scales, binSizes(j) * ones(1,size(myframes,2)) / imsizes(j)];
    end

    s.width = width;
    s.height = height;
    s.scales = scales;

    s.feaArr = double(descs);
    s.x = frames(1,:);
    s.y = frames(2,:);

    beta = LLC_pooling_fast(s, codebook, pyramid, knn, flann_options);

    beta = sparse(beta);
