## Experiments

- Run the fusion of classes and features classification experiments
    - Output pascal metaclass classifiers as features

- Evaluate caffe-based classifier performance

- Look at Flickr tags search results

## Ideas

- Equalize dataset by content to get less biased style performance numbers

## Fixes

- never pass pred_prefix into a function; it should be a canonical constant

- abstract the insertion of dataframes into mongo under vislab/collection, and make use of it in ui/app.py

- make sure name of every dataset df index is "image_id"

- a substantial number of flickr_80k images are now missing. should update dataset, or at least exclude the missing ones? can tell which are missing using the convert_gif function in bashrc

## Next

- implement image annotation UI

- make caffe-based classifier module using JL's python bindings

- make downloading and resizing images a job-queue based thing

- Large-scale style-filtered search demo:
    - compute fc6 features on a large set of pinterest images that don't overlap with my training set
    - compute fc6 features on a large set of flickr images for some tag query

- Generate Sphinx autodoc and display it on my doc page

- Compute pairwise distances for images for different features

- Features
    - whitened HOG feature (wrap Jon's code)
    - face detection
    - portrait and landscape classifiers: add to the PASCAL meta-class classifiers
    - make featurization server: gets filename, outputs result back on queue
    - implement presence of text classifier

- could be good figure: sort imagenet images for a few categories by beauty

- to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature

Similarity demo
    - add paintings data
    - add another level of decaf feature
    - add WHOG feature
    - re-implement the single-similarity page in addition to per-style similarity page
    - mode to sort images by proximity in weighted or unweighted feature space (need weights)

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
