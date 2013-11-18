## Next
Style transfer
    - use the gradient at data level of the convnet to apply to the pixels. will need to regularize

Features
    - whitened HOG feature (wrap Jon's code)
    - face detection
    - portrait and landscape classifiers: add to the PASCAL meta-class classifiers
    - make featurization server: gets filename, outputs result back on queue
    - implement feature computation in vislab that computes multiple levels of decaf feature at once
    - implement presence of text classifier
    - local features: I guess like Alyosha's stuff

Classification experiments
    - store bit_precision in the results table somewhere.

Single-person classification accuracy:
    - how good is a single person at predicting the style classification task? the ava task?

VW:
    - get vector of weights from VW.
        : in fact, need --inverse_hash to do this properly, and that seems to slow things down substantially (haven't tried though)
    - revamp cross-validation to write all results to database to allow iterative improvement and visualizing the effects of the parameters
    - implement a dot-product distance computation mode in vowpal wabbit that outputs all pairwise distances for the data points; if a model is provided, the distances are in the projected space. this will also implement the k-nn classifier.

Evaluation
    - average over multiple random subsets of test-balanced data

Datasets
    - make memorability/interestingness dataset interface (aude's data with extra interestingness scores)
    - store widths and heights in database for better layout on client
    - weight AVA examples according to inverse distance to mean (should work better than the delta method)

Results analysis
    - nice force-based layout graph of confusion matrix
    - could be a good figure: sort imagenet images for a few categories by beauty
    - aaron's idea to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature

Image search demo
    - get tags for the flickr dataset

Similarity demo
    - add paintings data
    - add another level of decaf feature
    - add WHOG feature
    - re-implement the single-similarity page in addition to per-style similarity page
    - mode to sort images by proximity in weighted or unweighted feature space (need weights)

New image demo
    - make a demo where a brand-new image is processed with the decaf feature, and analyzed with beauty and style classifiers.

Job queue system
    - launch workers separately from the script that submits jobs for them, but with their own script
    - be able to kill workers with scancel
    - report when workers get killed

Misc
    - need util function for syncing up dataframe with mongodb collection. useful for datasets.
    - graphically improve the image page table: say TP, FP, TN, FN, and color true/false with green/red as well
        :: Think of this as a general thing: want to be able to format tables with color depending on parameters, in Javacript. Publish blog post on the solution.
    - make diagram of data sources and labels and features and publish on blog

Recommendations
    - analyze it and try to form prediction dataset
        - how many images does a user like on average?
        - how many users favorited the average image?
        - what is the overlap between users? per image?

Bugs to fix:
    - preds_panel seems to contain objects instead of floats! yet sorting and comparisons still work fine. get a handle on this.

## Done

(cleared nov 15)
