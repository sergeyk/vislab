## Doing

- PICK UP AT:
Think I finally figured out:
- `predict_task` is `clf` or `regr`
- `predict_label` is a string with wildcard expansion that matches column names.
    if multiple columns are matched, then classification is done in the multi-class, OAA style.
    if a single column is matched, then classification is done in the binary style.

- PICK UP AT: untangle the dataset loading stuff.... the initial load is done in the dataset-specific module. the load from cache is done from dataset. so no force in dataset. then remove all duplicate logic for constructing datasets for prediction from ava and dataset.

- put this list into a google doc where i can highlight lines with different colors. go through and higlight items by importance: "must be done for draft", "must be done for final", "nice to have", "not important". bold items that are currently being worked on.

- get ava dataset up to par
    - add rating_mean_bin and rating_mean_norm_bin labels
    - load style information into AVA by default
    - get rid of the url map thing

- update dataset stats notebooks for
    - ava
    - flickr
    x pascal
    x wikipaintings

- train pascal content classifiers
    x merge into superclasses
    - use 2012: currently downloading to /u/vis/x1/common/PASCAL
    - launch feature computation

- get VW oaa working, with tests

## Next

Prediction
- fix up the cmd interface: no splitting of a flag
- compute performance on AVA
- compute performance with the conv5 feature to compare
- compute performance with convnet-retrained feature
- move predict runtime into vislab/predict.py

VW:
- implement oaa mode for vw
- add tests for vw
- call classifier/regression metrics in results module from vw
- get vector of weights from VW. (vowpal_porpoise doesnt seem to do this)
    : in fact, need --inverse_hash to do this properly, and that seems to slow things down substantially (haven't tried though)
- expand parameter space with learning rate
- revamp cross-validation to write all results to database to allow iterative improvement and visualizing the effects of the parameters

Datasets
- move the map urls task to ava dataset file, shouldn't be in run
- update get_url to use db instead of loading dataframe
- make memorability/interestingness dataset interface (aude's data with extra interestingness scores)
- increase Flickr style dataset size to 100K
- store widths and heights in database for awesome layout on client
- weight AVA examples according to inverse distance to mean (should work better than the delta method)

Features
- add step to cache features to h5 and vw at the end of a feature computation run
- move feature runtime stuff into aphrodite/features.py
- make featurization server: gets filename, outputs result back on queue
- implement WHOG
- compute WHOG on all datasets
- implement Labhist in Python
- compute Labhist on all datasets
- implement feature computation in vislab that computes multiple levels of decaf feature at once
- implement using classifier output as a feature
- compute the pascal metaclass feature for all datasets
- do additional training of the convnet with the style images
- implement portrait vs landscape classifier
- implement presence of text classifier

Analysis
- plot correlation matrix between objects and styles
- force-based layout graph of confusion matrix
- could be a good figure: sort imagenet images for each category by beauty
- aaron's idea to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature
-  see examples where Deep Learning works, and other features fail. For example, it seems like color histograms and object recognition ought to work in a lot of cases. And, indeed, they do. What is Deep Learning doing that the other features aren't?

Similarity
- get similarity display for paintings
- add another level of decaf feature for similarity
- add a WHOG feature for straight-up shape similarity
- re-implement the single-similarity page in addition to per-style similarity page
- mode to sort images by proximity in weighted or unweighted feature space (need weights)

Cluster upkeep
- remove features from mongo and cache to blosc-compressed h5 or iopro csv
- aggregate all the downloaded AVA photos from all the machines onto /u/vis/x1
- launch workers separately from the script that submits jobs for them, but with their own script
    - be able to kill workers with scancel
    - report when workers get killed

Search
- make a tag-based search interface

Other demos
- make demo where a brand-new image is processed with the decaf feature, and analyzed for beauty, and with style classifiers.

Misc
- config.json shouldn't be in the git repo; config.json.example instead
- need util function for syncing up dataframe with mongodb collection. useful for datasets.
- graphically improve the image page table: say TP, FP, TN, FN, and color true/false with green/red as well
    :: Think of this as a general thing: want to be able to format tables with color depending on parameters, in Javacript. Publish blog post on the solution.
- implement a dot-product distance computation mode in vowpal wabbit that outputs all pairwise distances for the data points; if a model is provided, the distances are in the projected space. this will also implement the k-nn classifier.
- make diagram of data sources and labels and features and publish on blog

Recommendations
- move fav_user_ids feature to own file for personal recommendations in vislab
- analyze it and try to form prediction dataset
    - how many images does a user like on average?
    - how many users favorited the average image?
    - what is the overlap between users? per image?

## Ideas

- "Similar image, but less beautiful."
- "Similar image, but more hazy."

- similarity ratings interface: this image is closer than this one

- idea: can introduce a third label, NOTSURE, which means that the image should simply not be a part of the training/test set for the concept. this is the label that would be set by the UI to clean up data.

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
