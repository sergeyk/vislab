## Doing

- add rating_mean_bin and rating_mean_norm_bin labels

- load style information into AVA by default

## Next

- need util function for syncing up dataframe with mongodb collection. useful for datasets.

- re-implement data exploration view in vislab and get it working for all datasets

- each dataset should have a notebook page showing some plots and such

- idea: deep learning for telling apart artistic style

- get rid of the url map thing for AVA dataset

- abstract classifier/regression metrics into own module, and call them from vw.

- plot confusion matrix for the wikipaintings styles and send email to jessica
    - idea: confusion matrix plotted vs confidence in the time dimension

- do I need to use the same test set for all classes?

- compute VW classifier performance with the fc6_flatten feature

- what's the order of features in the wikipaintings dataset?

- implement oaa mode for vw

- add tests for vw

- add tests for wikipaintings

- re-implement the single-similarity page in addition to per-style similarity page

- implement data approval view

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

## Notes

- http://0.0.0.0:4000/similar_to/f_9209451148
    - black and white image of a girl. doesn't find bright-energetic features

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
