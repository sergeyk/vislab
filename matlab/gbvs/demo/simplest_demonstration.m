
% example of how to call gbvs with default params

img = imread('samplepics/4.jpg');
out_gbvs = gbvs(img);
out_itti = ittikochmap(img);


subplot(2,3,1);
imshow(img);
title('Original Image');

subplot(2,3,2);
show_imgnmap(img,out_gbvs);
title('GBVS map overlayed');

subplot(2,3,3);
show_imgnmap(img,out_itti);
title('Itti/Koch map overlayed');


subplot(2,3,5);
imshow( out_gbvs.master_map_resized );
title('GBVS map');

subplot(2,3,6);
imshow(out_itti.master_map_resized);
title('Itti/Koch map');

