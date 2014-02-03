function rimg = rankimg(img)

img = uint8(mat2gray(img)*255);

rimg = zeros(size(img));

img = img(:);

for i = 0 : 255
  rimg( img == i ) = mean( img < i );
end