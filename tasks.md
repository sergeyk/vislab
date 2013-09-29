## Doing

- update get_url to use db instead of loading dataframe

- get classifier scores for the painting classification

- get similarity display for paintings

- experiment with standardizing the features and not using negative values

- add another level of decaf feature for similarity

- add a HOG feature for straight-up shape similarity

- implement feature computation in vislab that computed multiple levels of decaf feature

## Quick

- re-implement data exploration view in vislab

- add tests for wikipaintings

- re-implement the single-similarity page in addition to per-style similarity page

## Long

- implement data approval view


## Ideas

- "Similar image, but less beautiful."
- "Similar image, but more hazy."

- similarity ratings interface: this image is closer than this one

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
