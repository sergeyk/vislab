function maps = get_maps(image_filenames, output_filename)

% Run from gbvs/ directory.
pathroot = pwd;
save -mat util/mypath.mat pathroot
addpath(genpath( pathroot ), '-begin');

for i=1:length(image_filenames)
    image_filename = image_filenames{i};
    current_img = imread(image_filename);
    current_img = imresize(current_img, [256, 256]);
    feat = gbvs(current_img);
    %figure,imagesc(feat.master_map);
    maps(i,:) = feat.master_map(:);
end

save(output_filename, 'maps');

