function p = makeGBVSParams()
 
p = {};

%%%%%%%%%%%%% general  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p.salmapmaxsize = 32;             % size of output saliency maps (maximum dimension)
                                  % don't set this too high (e.g., >60)
                                  % if you want a saliency map at the
                                  % original image size, just used rescaled
                                  % saliency map
                                  % (out.master_map_resized in gbvs())

p.verbose = 0;                    % turn status messages on (1) / off (0)
p.verboseout = 'screen';          % = 'screen' to echo messages to screen
                                  % = 'myfile.txt' to echo messages to file                                   

p.saveInputImage = 0;             % save input image in output struct
                                  % (can be convenient, but is wasteful
                                  %  to store uncompressed image data
                                  %  around)

p.blurfrac = 0.02;                % final blur to apply to master saliency map
                                  % (in standard deviations of gaussian kernel,
                                  %  expressed as fraction of image width)
                                  % Note: use value 0 to turn off this feature.

%%%%%%%%%%%%% feature channel parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

p.channels = 'DIO';               % feature channels to use encoded as a string
                                  % these are available:
                                  %   C is for Color
                                  %   I is for Intensity
                                  %   O is for Orientation
                                  %   R is for contRast
                                  %   F is for Flicker
                                  %   M is for Motion
                                  %   D is for DKL Color (Derrington Krauskopf Lennie) ==
                                  %     much better than C channel
                                  % e.g., 'IR' would be only intensity and
                                  %       contrast, or
                                  % 'CIO' would be only color,int.,ori. (standard)
                                  % 'CIOR' uses col,int,ori, and contrast

p.colorWeight = 1;                % weights of feature channels (do not need to sum to 1). 
p.intensityWeight = 1;             
p.orientationWeight = 1;
p.contrastWeight = 1;
p.flickerWeight = 1;
p.motionWeight = 1;
p.dklcolorWeight = 1;

p.gaborangles = [ 0 45 90 135 ];  % angles of gabor filters
p.contrastwidth = .1;             % fraction of image width = length of square side over which luminance variance is 
                                  % computed for 'contrast' feature map
                                  % LARGER values will give SMOOTHER
                                  %   contrast maps

p.flickerNewFrameWt = 1;          % (should be between 0.0 and 1.0)
                                  % The flicker channel is the abs() difference
                                  % between the *previous frame estimate* and
                                  % current frame.
                                  % This parameter is the weight used
                                  % to update the previous frame estimate. 
                                  % 1 == set previous frame to current
                                  %      frame
                                  % w == set previous frame to w * present
                                  %      + (1-w) * previous estimate
                                  
p.motionAngles = [ 0 45 90 135 ]; 
                                  % directions of motion for motion channel
                                  %  --> 0 , /^ 45 , |^ 90 , ^\ 135 , etc. 
                                  % question: should use more directions?
                                  % e.g., 180, 225, 270, 315, ?
                                  
%%%%%%%%%%%%% GBVS parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                  
p.unCenterBias = 0;               % turned off (0) by default. Attempts to undo some emergent
                                  % center bias in GBVS (by pointwise-multiplying final saliency map by 
                                  % an inverse-bias map).

p.levels = [2 3 4];               % resolution of feature maps relative to original image (in 2^-(n-1) fractions)
                                  % (default [ 2 3 4]) .. maximum level 9 is allowed
                                  % these feature map levels will be used
                                  % if graph-based activation is used.
                                  % otherwise, the ittiCenter/Delta levels
                                  % are (see below)
                                  % minimum value allowed  = 2
                                  % maximum value allowed  = 9

p.multilevels = [];               % [1 2] corresponds to 2 additional node lattices ,
                                  % ... one at half and one at quarter size
                                  % use [] for single-resolution version of algorithm.

p.sigma_frac_act = 0.15;          % sigma parameter in activation step of GBVS (as a fraction of image width) - default .15
p.sigma_frac_norm = 0.06;         % sigma parameter in normalizaiton step of GBVS (as a fraction of image width) - default .06
p.num_norm_iters = 1;             % number of normalization iterations in GBVS - default 1

p.tol = .0001;                    % tol controls a stopping rule on the computation of the equilibrium distribution (principal eigenvector)
                                  % the higher it is, the faster the algorithm runs, but the more approximate it becomes.
                                  % it is used by algsrc/principalEigenvectorRaw.m - default .0001
                                  

p.cyclic_type = 2;                % this should *not* be changed (non-cyclic boundary rules)
                                  
%%%%%%%%%% Parameters to use Itti/Koch and/or Simpler Saliency Algorith %%%%
                                  
                                  
p.useIttiKochInsteadOfGBVS = 0;   % use value '0' for Graph-Based Visual Saliency
                                  % use value '1' for Itti Koch algorithm: 
                                  % "A Model of Saliency-Based Visual
                                  % Attention for Rapid Scene Analysis",
                                  % PAMI 1998
                                  
p.activationType = 1;             % 1 = graph-based activation (default)
                                  % 2 = center-surround activation (given
                                  %     by ittiCenter/DeltaLevels below)
                                  % ( type 2 used if useIttiKoch..= 1 )

p.normalizationType = 1;          % 1 = simplest & fastest. raises map values to a power before adding them together (default)
                                  % 2 = graph-based normalization scheme (no longer recommended)
                                  % 3 = normalization by (M-m)^2, where M =
                                  %     global maximum. m = avg. of local
                                  %     maxima
                                  % ( type 3 used if useIttiKoch..=1 )

p.normalizeTopChannelMaps = 0;    % this specifies whether to normalize the 
                                  % top-level feature map of each
                                  % channel... (in addition to normalizing
                                  % maps across scales within a channel)
                                  % 0 = don't do it (default)
                                  % 1 = do it. (used by ittiKoch scheme)
                                  
p.ittiCenterLevels = [ 2 3 ];     % the 'c' scales for 'center' maps
                                   
p.ittiDeltaLevels = [ 2 ];        %  the 'delta' in s=c+delta levels for 'surround' scales
                                  %  NOTE: for more faithful implementation , use deltaLevels = [ 2 3 ], not [ 2 ] 
                                  %  however, 3 can be problematic for images < 640 in width or height

p.ittiblurfrac = 0.03;            % apply final blur to master saliency map
                                  % (not in original Itti/Koch algo. but improves eye-movement predctions)