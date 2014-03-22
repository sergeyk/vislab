## Fixes

## Today

- Merge dev to master
- Compute fc6 features on flickr
- Cache fc6 on flickr to vw
- Compute fc6 features on wikipaintings
- Cache fc6 on wikipaintings to vw
- Compute fc6 features on pinterest
- Cache fc6 on pinterest to vw
- Compute fc7 features on flickr
- Cache fc7 on flickr to vw
- Compute fc7 features on wikipaintings
- Cache fc7 on wikipaintings to vw
- Compute fc7 features on pinterest
- Cache fc7 on pinterest to vw


## Tomorrow

- Form list of experiments that need to be run, rank by importance

## Next

- make sure Helen send me her Flickr-scraping code, so that I can integrate it into Vislab, and run it again to update the new dataset with tags

Single-person classification accuracy:
    - how good is a single person at predicting the style classification task? the ava task?

- Replace classifier: VW -> caffe
    - In feature.py, output to several HDF5 files (of max size 2GB) instead of just one.
    - Modify Caffe to be able to take label file as separate from feature file.
    - Modify Caffe to be able to take multiple feature files (should be easy, as separate layers).

- Generate Sphinx autodoc and display it on my doc page

- Compute pairwise distances for images for different features

- Features
    - whitened HOG feature (wrap Jon's code)
    - face detection
    - portrait and landscape classifiers: add to the PASCAL meta-class classifiers
    - make featurization server: gets filename, outputs result back on queue
    - implement presence of text classifier

Evaluation
    - average over multiple random subsets of test-balanced data

Datasets
    - make memorability/interestingness dataset interface (aude's data with extra interestingness scores)
    - weight AVA examples according to inverse distance to mean (should work better than the delta method)

Results analysis
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
    - be able to kill workers with scancel
    - report when workers get killed

- make the results server use Mongo (and then can populate dataframes from filtered results there)

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

## Done

x every occurrence of vislab.repo_dirname should be replaced by path in vislab.config
x publish website for documentation, from docs/ directory
x set the redis and mongodb server hostname and port in config file
x put Adobe copyright text in relevant files
x integrate feature.py and _feature.py: right now, messy nesting
x Replace convnet feature computation: decaf -> caffe
x Expand Flickr set to 5000 examples for 20 different styles (added Bokeh. Detailed, Texture).
x integrate Pinsplorer code into Vislab
x store image and page urls in the pinterest dataset: rsync the notebook from elbow first, don't want to re-write that code
x also, give pinterest dataset names exactly like the Flickr dataset
