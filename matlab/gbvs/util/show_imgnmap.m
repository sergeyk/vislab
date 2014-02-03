function hm = show_imgnmap( img , out )
hm = heatmap_overlay( img , out.master_map_resized );
imshow( hm );
