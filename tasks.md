## Experiments that need to happen for CVPR

oct 28
- [x] write out gbvs_saliency results for ava_style: bug in cache_to_h5
- [x] gbvs experiment on ava_style
- [x] get artist experiments results to load: fix some bug
    : now getting updated with validation data
- [x] assemble all trainval images from Flickr and Wikipaintings into rolled-out 3x256x256 vectors, and assign int labels to all of them. ping jeff
- [x] launch pascal clf training again -- can't find models. not OAA, don't want to mess with evaluation code that I'd have to change.
- [x] aesthetic prediction results: test locally, launch on cluster overnight

oct 29
- [x] re-launch aesthetic pred. results with increased memory, this is BS
- [x] launch finetuning training on durian (launched oct 29 1400)
- [x] run pascal content classifiers on all datasets
    : launched oct29 1800 (collection names are given in run_experiment.py)
    - [x] sync up all pascal models to flapjack and then push them out to all machines
- [x] sync up results on all datasets such that local mongo has up to date info
    - [ ] load aesthetic results

- [ ] develop method for loading the result pred_dfs for multiple classes and writing them back out as features
    PICK UP HERE: gotta match the order of decaf feature
- [ ] test and launch quadratic expansion with content classifiers experiments
- [ ] generate the content-style co-occurrence matrix
- [ ] get image similarity website going again with all features and datasets

Features
- whitened HOG feature (wrap Jon's code) [.5 day]
- portrait and landscape classifiers [.25 day]

Datasets
- add 2 more styles to Flickr set for total of 20

Experiments
- store bit_precision in the results table somewhere. can definetely influence performance!

Image search demo
    - get flickr tags for the flickr set

Single-person classification accuracy:
    - how good is a single person at predicting the style classification task? the ava task?

Style features
    - try style features for the ava aesthetic prediction task
    - for memorability task

Memorability dataset:
    - write interface
    - compute features, including style features
    - run experiments

## Next

VW:
- get vector of weights from VW. (vowpal_porpoise doesnt seem to do this)
    : in fact, need --inverse_hash to do this properly, and that seems to slow things down substantially (haven't tried though)
- expand parameter space with initial learning rate
- revamp cross-validation to write all results to database to allow iterative improvement and visualizing the effects of the parameters

Datasets
- make memorability/interestingness dataset interface (aude's data with extra interestingness scores)
- increase Flickr style dataset size to 100K
- store widths and heights in database for awesome layout on client
- weight AVA examples according to inverse distance to mean (should work better than the delta method)

Features
- make featurization server: gets filename, outputs result back on queue
- implement feature computation in vislab that computes multiple levels of decaf feature at once
- implement presence of text classifier
- can further improve the url

Analysis
- plot correlation matrix between objects and styles
- force-based layout graph of confusion matrix
- could be a good figure: sort imagenet images for each category by beauty
- aaron's idea to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature
-  see examples where Deep Learning works, and other features fail. For example, it seems like color histograms and object recognition ought to work in a lot of cases. What is Deep Learning doing that the other features aren't?

Similarity
- get similarity display for paintings
- add another level of decaf feature for similarity
- add a WHOG feature for straight-up shape similarity
- re-implement the single-similarity page in addition to per-style similarity page
- mode to sort images by proximity in weighted or unweighted feature space (need weights)

Cluster upkeep
- launch workers separately from the script that submits jobs for them, but with their own script
    - be able to kill workers with scancel
    - report when workers get killed

Other demos
- make demo where a brand-new image is processed with the decaf feature, and analyzed for beauty, and with style classifiers.

Misc
- need util function for syncing up dataframe with mongodb collection. useful for datasets.
- graphically improve the image page table: say TP, FP, TN, FN, and color true/false with green/red as well
    :: Think of this as a general thing: want to be able to format tables with color depending on parameters, in Javacript. Publish blog post on the solution.
- implement a dot-product distance computation mode in vowpal wabbit that outputs all pairwise distances for the data points; if a model is provided, the distances are in the projected space. this will also implement the k-nn classifier.
- make diagram of data sources and labels and features and publish on blog

Recommendations
- analyze it and try to form prediction dataset
    - how many images does a user like on average?
    - how many users favorited the average image?
    - what is the overlap between users? per image?

## Ideas

- can introduce a third label, NOTSURE, which means that the image should simply not be a part of the training/test set for the concept. this is the label that would be set by the UI to clean up data.

## Done

- link to go to a random image (sep 12)
- handle multiple sources of features: add the style classifier feature first (sep 13)
- display top 8 results for all styles on one page (sep 13)
- start detailed data collection on cluster (sep 23)
- switch to using a standalone queue-based similarity engine (sep 24)
    : rq is too rigidly dependent on pickled function calls.
    : beanstalkd doesn't have results store.
    x roll own solution, based on redis.
    x get rid of using eval: register the instance method instead
- switch to using mongo instead of dataframe for the data explorer task (sep 24)
- plot correlation matrix between style and genre
    - dust off the correlation matrix code from pascal days
- abstract classifier/regression metrics into own module
- plot confusion matrix for the wikipaintings styles
- top-k accuracy multi-class metric
- plot AP scores in the balanced whole-set setting from multiclass metrics
x multiclass classifier metrics
    x confusion table
    x plot P-R curves for all classes
x call classifier/regression metrics in results module from vw
x train pascal content classifiers
- fix up the cmd interface: no splitting of a flag
- move predict runtime into vislab/predict.py
- move the map urls task to ava dataset file, shouldn't be in run
- move feature runtime stuff into aphrodite/features.py
- get image url stuff working robustly for all datasets by adding dataset name to the calling chain. OR: should the feature pipeline be rethought to get the filenames instead of ids?
- move fav_user_ids feature to own file for personal recommendations in vislab
- config.json shouldn't be in the git repo; config.json.example instead
x re-run classification and regression with decaf_fc6 feature on style subset
- implement across-feature metric comparison: like the AVA style AP barplot
- run classification on ava style with decaf_fc6 feature
- Plot the chance performance on the top-k accuracy plot
- AP barplot
    - be able to display multiple features
    - add "chance" bar: generate random confidences and see how well that works
- decaf AVA style results have lower AP than published, and lab_hist is even worse
    x form dataset respecting the original train/test split
        x re-running experiments, launched oct 17 2145
             - still bad!
    - the confounding thing is that lab_hist performs even worse: super bad! but in the paper, it's reported to be super good.
        - experiment with sklearn classifiers and lab_hist feature to see what the performance is. why is it so bad?
            - if it is still bad, re-compute the lab hist features and don't standardize them to make them sparse!
    - shuffle order of features in the vw text file
        - confirming that the order is bad: positive examples are clumped together
    - better cross-validation, more passes
- get multiclass prediction dataset generator working
- add vw tests
- implement and test vw oaa mode
- compare vw oaa mode results to using separate binary classifiers
- One method for outputting all evaluations of a prediction experiment. takes a collection name, a label_df, and a list of features to compare.
- make the colors used in the ap bar plot and the top k plot be the same (order the columns the same way)
- plot per-feature confusion matrices in a visible place: just print less from mc_metrics_feat_comparison, and title images and should be good
- Adding Murray-CVPR-2012 results:
    mc_metrics['ap_df']['Murray-CVPR-2012'] = 0
    mc_metrics['ap_df']['Murray-CVPR-2012'].iloc[:-1] = [
        .44, .51, .64, .74, .73, .43, .50, .40, .69, .30, .48, .72, .39, .57]
    mc_metrics['ap_df']['Murray-CVPR-2012'].iloc[-1] = mc_metrics['ap_df']['Murray-CVPR-2012'].iloc[:-1].mean()
- debug mc_bit feature: seems to always be true. then re-run mc_bit computation
- amend the cache_to_h5 thing that if the feature sums to 1 across rows, don't standardize (for histograms)
- save the results and predictions panel just to filesystem in common space instead of database. size limit...
