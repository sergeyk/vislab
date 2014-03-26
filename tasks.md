## Ideas

- Can dataset be equalized by content (to get unbiased style performance numbers)

## Fixes

## Next

- UI
    - make the results server use Mongo (and then can populate dataframes from filtered results there)
    - add basic results view to current UI
    - add advanced results view: `source_dataset` support and so on
    - restart data view server and launch on flapjack

- Run the fusion of classes and features classification experiments
    - Output pascal metaclass classifiers as features

- Large-scale style-filtered search demo:
    - compute fc6 features on a large set of pinterest images that don't overlap with my training set
    - compute fc6 features on a large set of flickr interesting images

Human classification accuracy:
    - how good is a single person at predicting the style classification task? (Trent is running this on mech turk)

- Replace classifier: VW -> caffe
    - In feature.py, output to several HDF5 files (of max size 2GB) instead of just one.
    - Modify Caffe to be able to take label file as separate from feature file.
    - Modify Caffe to be able to take multiple feature files (should be easy, as separate layers)
    - Be able to call caffe directly from Python

- Generate Sphinx autodoc and display it on my doc page

- Compute pairwise distances for images for different features

- Features
    - whitened HOG feature (wrap Jon's code)
    - face detection
    - portrait and landscape classifiers: add to the PASCAL meta-class classifiers
    - make featurization server: gets filename, outputs result back on queue
    - implement presence of text classifier

- make memorability/interestingness dataset interface (aude's data with extra interestingness scores)

- could be good figure: sort imagenet images for a few categories by beauty

- to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature

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
    - `map_through_rq` should keep track of succeeding jobs, not workers that it launched


Misc
    - need util function for syncing up dataframe with mongodb collection. useful for datasets.
    - graphically improve the image page table: say TP, FP, TN, FN, and color true/false with green/red as well
        :: Think of this as a general thing: want to be able to format tables with color depending on parameters, in Javacript. Publish blog post on the solution.
    - make diagram of data sources and labels and features and publish on blog

Image Recommendations
    - analyze it and try to form prediction dataset
        - how many images does a user like on average?
        - how many users favorited the average image?
        - what is the overlap between users? per image?

Bugs to fix:
    - preds_panel seems to contain objects instead of floats! yet sorting and comparisons still work fine. get a handle on this.
