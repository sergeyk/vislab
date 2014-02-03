
Graph-Based Visual Saliency (MATLAB source code)
http://www.klab.caltech.edu/~harel/share/gbvs.php

Jonathan Harel
jonharel@gmail.com
California Institute of Technology

========================================================================================

This is an installation and general help file for the saliency map MATLAB code here.

========================================================================================

What you can do with this code:

(1) Compute a "Graph-Based Visual Saliency" map for an image or image sequence (video)
    (as described in J. Harel, C. Koch, and P. Perona. "Graph-Based Visual Saliency", 
    NIPS 2006
    http://www.klab.caltech.edu/~harel/pubs/gbvs_nips.pdf)

(2) Compute the standard Itti, Koch, Niebur (PAMI 1998) saliency map.

(3) Compute modified versions of the above by altering the input parameters.

========================================================================================

Step-by-step start-up procedure:

(1) Add gbvs to your path:

    Change into the directory containing this file, and enter at the matlab prompt:

     >> gbvs_install

    If you are on a shared machine, you may get an error message such as:

      Warning: Unable to save path to file '/opt/matlab/toolbox/local/pathdef.m'
       In savepath at 162
       In gbvs_install at 5

    In that case, comment out the savepath (i.e., 'savepath' => '% savepath') 
    command in gbvs_install.m, and add this line to your startup.m file:

    run ???/gbvs_install

    where "???" is replaced by the main gbvs/ directory, which contains the
    gbvs_install function

(2) Now you are ready to compute GBVS maps:

   Demonstrations:

      >> simplest_demonstration

     see demo/demonstration.m for more complicated demo or run:
     [Note: if you get an error, see point (3) below]
     
      >> demonstration

   Basic Usage Example:

     >> out = gbvs( 'samplepics/1.jpg' );

   You can also compute an Itti/Koch map as follows:
     
     >> out = ittikochmap( 'samplepics/1.jpg' );

   Or, to call GBVS simplified to some extent (e.g. no Orientation channel) so that it runs faster, use

     >> out = gbvs_fast( 'samplepics/1.jpg');

   Now, out.master_map contains your saliency map, and out.master_map_resized is
   this saliency map interpolated (bicubic) to the resolution of the original 
   image.

   For video (not static images):
    You need to pass into gbvs() previous frame information, which is returned
    on output at every call to gbvs(). 
   
   See demo/flicker_motion_demo.m

   Here is the heart of it:

    motinfo = [];   % previous frame information, initialized to empty
    for i = 1 : N
      [out{i} motinfo] = gbvs( fname{i}, param , motinfo );
    end

(3) If you are not on 32 or 64 bit Windows, or on Intel-based Mac, or 32 or 64 bit Linux,
    and calling simplest_demonstration results in an error, you may have to compile
    a few .cc source code files into binary "mex" format.

    You can do that as follows. From the gbvs/ directory, in matlab, run:

     >> gbvs_compile

    If this works properly, there should be no output at all, and you're done!
    Then go back to step (2), i.e. try running the demonstration.

    Error note:
      If this is your first time compiling mex files, you may have to run:

        >> mex -setup

      and follow the instructions (typically, enter a number, to select a co-
      mpiler. then you can run "gbvs_compile"; if it doesn't work, run 
      "mex -setup" again to select a different compiler, run "gbvs_compile" 
      again, etc.)

========================================================================================

Helpful Notes:

(1) inputs of gbvs():

     * the first argument to gbvs() can be an image name or image array
     * there is an optional, second, parameters argument

(2) outputs of gbvs():

     * all put into a single structure with various descriptive fields.
     * the GBVS map: master_map
      (interpolated to the resolution of the input image: master_map_resized)
     * master saliency map for each channel: feat_maps (and their names, 
       map_types)
     * all intermediate maps to create the previous two (intermed_maps). see 
      gbvs.m for details

(3) the parameter argument:

     * initialized by makeGBVSParams.m -- read that for details.

      Some very sparse notes on fields of the parameter argument:

        sigma_frac_act    controls the spatial spread of the function modulating 
                          weights between different image locations (in image widths).
                          greater value means greater connectivity between distant
                          locations.

        tol               tolerance parameter. governs how accurately the princi-
                          pal eigenvector calculation is performed. change it to 
                          higher values to make things run faster.

        levels            the resolution of the feature maps used to compute the
                          final master map, relative to the original image size
                                                    
(4) Notes on feature maps:

     * are produced by util/getFeatureMaps.m

     * by default, color, intensity, orientation  maps are computed.

       which channels are used is controlled by the parameters argument. in part-
       icular, you can choose which of these is included by editing the 
       params.channels string (see makeGBVSParams.m). you can set
       their relative weighting also in the parameters.

       If you want to introduce a new feature channel, put a new function into 
       util/featureChannels/ . Make sure to edit the channels string appropria-
       tely. Follow pattern of other channels for proper implementation.

(5) If you want to compare saliency maps to fixations (e.g., inferred from
       scanpaths recorded by an eye-tracker), use:

         >> score = rocScoreSaliencyVsFixations(salmap,X,Y,origimgsize)

        This outputs ROC Area-Under-Curve Score between a saliency map and fixat-
        ions.

         salmap       : a saliency map
         X            : vector of X locations of fixations in original image
         Y            : vector of Y locations of fixations in original image
         origimgsize  : size of original image (should have same aspect ratio as
                        saliency map)

========================================================================================

Credits:

(1) saltoolbox/ directory -- adapted from: Dirk Walther, http://www.saliencytoolbox.net

(2) Thanks to Alexander G. Huth for help with making heatmap_overlay.m readable.

========================================================================================

Revision History

first authored 8/31/2006
Revised 4/25/2008
Revised 6/5/2008
Revised 6/26/2008 
	added Itti/Koch algorithm
Revised 8/25/2008 
	added Flicker/Motion channels
Revised 11/3/2008 
	added myconv2
Revised 2/19/2010 
	added initcache to reduce initialization times
Revised 3/18/2010 
	added attenuateBordersGBVS to O_orientation call
Revised 1/17/2011 
	added attenuateBordersGBVS to master_map. 
	changed boundary condition in padImage 
	changed ittiDeltaLevels for ittiKoch to just [2] by default
	removed Intensity channel from gbvs_fast	
Revised 10/24/2011
	added unCenterBias to parameters, turned it on by default
Revised 7/24/2012
	show_imgnmap returns output. for win users: initGBVS uses fullfile.
