## Doing

- get ava dataset up to par
    - add rating_mean_bin and rating_mean_norm_bin labels
    - load style information into AVA by default
    - get rid of the url map thing
- update dataset stats page for aphrodite, flickr, pascal
- train pascal content classifiers: merge into superclasses
- get VW oaa working

## Next

- config.json shouldn't be in the git repo; config.json.example instead

- PICK UP AT: re-run AVA experiments: get ava in order, find out what my AP #s are.

- make memorability/interestingness dataset interface (aude's data with extra interestingness scores)

- test on memorability dataset

- multiclass classifier metrics
    x confusion table
    - plot P-R curves for all classes

- force-based layout graph of confusion matrix

- ! plot correlation matrix between objects and styles

- classify for objects

- send email to jessica with paper draft

- get displays going again

- need util function for syncing up dataframe with mongodb collection. useful for datasets.

- re-implement data exploration view in vislab and get it working for all datasets

- idea: deep learning for telling apart artistic style

- call classifier/regression metrics in results module from vw

- compute VW classifier performance with the fc6_flatten feature

- what's the order of features in the wikipaintings dataset?

VW:
- implement oaa mode for vw
- add tests for vw

Wikipaintings:
- add tests for wikipaintings

- re-implement the single-similarity page in addition to per-style similarity page

- aaron's idea to analyze the deep feature: see if you can regress to the color histogram feature from the deep feature

- update get_url to use db instead of loading dataframe

- get classifier scores for the painting classification

- get similarity display for paintings

- experiment with standardizing the features and not using negative values

- add another level of decaf feature for similarity

- add a HOG feature for straight-up shape similarity

- implement feature computation in vislab that computed multiple levels of decaf feature

- load pascal and compute deep feature on it


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
