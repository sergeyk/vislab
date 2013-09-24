## Doing

- switch to using mongo instead of dataframe for the data explorer task

## Quick

- add tests for wikipaintings

- re-implement the single-similarity page in addition to per-style similarity page

## Long


- implement data approval view

- give more feature options: different levels of decaf feature
    : need to re-run feature computation on the images, saving multiple levels of features
- add a HOG feature for straight-up shape similarity

## Ideas

- "Similar image, but less beautiful."
- "Similar image, but more hazy."

## Notes

- http://0.0.0.0:4000/similar_to/f_9209451148
    - black and white image of a girl. doesn't find bright-energetic features

## Done

- link to go to a random image (sep 12)
- handle multiple sources of features: add the style classifier feature first (sep 13)
- display top 8 results for all styles on one page (sep 13)
- start detailed data collection on cluster (sep 23)
- switch to using a standalone queue-based similarity engine
    : rq is too rigidly dependent on pickled function calls.
    : beanstalkd doesn't have results store.
    x roll own solution, based on redis.
    x get rid of using eval: register the instance method instead
