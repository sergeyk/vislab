function output = my_rgb2gray(image)
%by James Hays
%I wrote this to avoid using the image processing toolbox

if(size(image,3) == 1)
   fprintf('warning, image already gray\n');
   output = image;
else
   output = .2989 * image(:,:,1) + .5870 * image(:,:,2) + .1140 * image(:,:,3);
end
