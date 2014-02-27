function [r,g,b,ii] = mygetrgb( img )

     r = img(:,:,1);
     g = img(:,:,2);
     b = img(:,:,3);
     ii = max(max(r,g),b);
