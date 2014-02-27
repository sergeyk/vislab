function feats = lab_hist(image_filenames, output_filename)
% Script to convert images into L*a*b* histogram features.
% Takes a list of image filenames.
% Returns a matrix of features.
% RUN FROM APHRODITE DIRECTORY

addpath('matlab/lab_histogram');

for i=1:length(image_filenames)
    image_filename = image_filenames{i};
    current_img = im2double(imread(image_filename));
    if (ndims(current_img) == 2)
        current_img = repmat(current_img, [1, 1, 3]);
    end

    [L, a, b, C, h] = RGB2LAB_and_LCh_with_gray_removal(current_img);

    CIELAB_hist = histnd(...
        [L a b], ...
        [-inf 55 70 90 inf], ...
        [-inf -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 inf], ...
        [-inf -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 inf]);
    CIELAB_hist = CIELAB_hist(1:4, 1:14, 1:14);
    feats(:, i) = CIELAB_hist(:) ./ sum(CIELAB_hist(:))
end
feats = feats';

save(output_filename, 'feats', '-ascii');
