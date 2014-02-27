function show_imgnmap2( img , smap )
smap = mat2gray( imresize(smap,[size(img,1) size(img,2)]) );
imshow(heatmap_overlay( img , smap ));