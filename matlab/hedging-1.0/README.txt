====================================================================
1. Overview
====================================================================

This code accompanies the paper

Jia Deng, Jonathan Krause, Alex Berg, and Li Fei-Fei. 
Hedging Your Bets: Optimizing Accuracy-Specificity Trade-offs in
Large-Scale Visual Recognition. CVPR 2012.

It implements the DARTS algorithm and reproduces the results on the
ILSVRC65 dataset, i.e. plots in Figure 5 of the paper. 

====================================================================
2. Quick run
====================================================================

The easier way to evaluate DARTS is to download the minimum set of
precomputed features and models from our website. Untar them and
overwrite the 'features' and 'models' folder. Then jump to section 6
of this document. 

====================================================================
3. Images
====================================================================

The images of ILSVRC65 should be placed in the 'images' folder. You
can download all images for ILSVRC65 from our website. There should be
the following files:

images/ilsvrc65.train.subset0.tar
images/ilsvrc65.train.subset1.tar
...
images/ilsvrc65.train.subset4.tar
images/ilsvrc65.val.tar
images/ilsvrc65.test.tar

There are 5 different training sets (subset 0, subset 1, ... subset
4). Note that there is no need to extract images from the tar files.


====================================================================
4. Features
====================================================================

You can skip this section by using the pre-computed features on our
website. Simply download them and place them into the 'features'
folder. 

Alternatively, You can convert images to features through the
following steps.

(1) obtain vlfeat-0.9.9 source code and place the folder
'vlfeat-0.9.9' under 'third-party'. Compile it for matlab. 

(2) run

      ./extract_features.sh test.input.tar test.llc.mat

It takes a tar file of images and output a matlab file containing the
features for all iamges. test.input.tar is a sample input for testing
the code. You can compare your test.llc.mat with the file
test.llc.reference.mat provided in the folder. The contents of two
files should be exactly the same. Note that it does not mean that the
two mat files are identical. You need to compare the data after
loading the files into matlab. 

If you get slightly different feature vectors, it could be that your
'convert' command is a different version. 

Below is our version of the convert command (run 'convert --version')

Version: ImageMagick 6.6.1-5 2010-06-09 Q16 http://www.imagemagick.org
Copyright: Copyright (C) 1999-2010 ImageMagick Studio LLC
Features: OpenMP 

If you don't reproduce the exact features, the performance of our
pre-trained classification models will be affected. But as long as you
train your own models, the classification performance should be similar. 

To extract features for training, test, and validation, run

./extract_features.sh ../images/ilsvrc65.train.subset0.tar ../features/ilsvrc65.train.subset0.llc.mat
./extract_features.sh ../images/ilsvrc65.train.subset1.tar ../features/ilsvrc65.train.subset1.llc.mat
        ...
./extract_features.sh ../images/ilsvrc65.test.tar ../features/ilsvrc65.test.llc.mat
./extract_features.sh ../images/ilsvrc65.val.tar ../features/ilsvrc65.val.llc.mat


====================================================================
5. Train flat models
====================================================================

You can skip this section by using the pre-trained models provided on
our website. Simply download them and place them in the 'models' folder.

Alternatively, you can train the models through the following steps.

(1) Compile liblinear (included in the 'third-party' folder)

(2) Compile platscaling.cpp by typing
      
       mex platscaling.cpp

(3) In Matlab run

       train_models

This will give you a SVM model trained with C=100 on training subset
0. It also includes SVM outputs from cross validation that are used
for probability calibration. The result is saved in 
'models/lsvrc65.subset0.C100.model.mat'. You can modify the script to 
train models on other training subsets or with other C parameters.


====================================================================
6. Run DARTS
====================================================================

If you have downloaded all precomputed features and models (or the
minimum set) and put them in the proper folders, you are ready to
start running DARTS. Otherwise, you need to make sure that the
features for training (subset 0), validation and test exist in
'features' and the model file 'lsvrc65.subset0.C100.model.mat' is in 
'models'. 

In Matlab run:

       run_DARTS

It will calibrate the probabilities and produce for DARTS as well as
some of the baselines. It saves the results into
'inf_results.mat'. If you use our pre-computed features and models,
the results should be the same as 'inf_results.reference.mat'.
You can modiefy the script to try DARTS on models trained on other
subsets and with different parameters. 

To make plots, run in Matlab:

       make_plots

It will generate the plots in Figure 5 of the paper except for the
Tree related curves. Note that the plots here are similar but not 
identical to those in Figure 5 because they are from only one training
subset. Figure 5 is from averaging 5 training subsets. 

