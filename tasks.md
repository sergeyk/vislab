## Fixes

## Today

x PROBLEM: cannot get vlg_extractor to work right now

- Make VW write its stuff to /tscratch when training, but copy all but the cache files to central location. this way, parallel learning can happen without overload disk, but we still are able to use source_dataset stuff

x Compute fc6 features on flickr
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc6 --dataset=flickr --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc6 --dataset=flickr && python vislab/feature.py cache_to_vw --features=caffe_fc6 --dataset=flickr

- Compute fc6 features on wikipaintings
    (technically, current results are fc6, so no need to recompute)
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc6 --dataset=wikipaintings --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc6 --dataset=wikipaintings &&     python vislab/feature.py cache_to_vw --features=caffe_fc6 --dataset=wikipaintings

x Compute fc6 features on pinterest
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc6 --dataset=pinterest_80k --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc6 --dataset=pinterest_80k &&     python vislab/feature.py cache_to_vw --features=caffe_fc6 --dataset=pinterest_80k

x Compute fc7 features on flickr
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc7 --dataset=flickr --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc7 --dataset=flickr && python vislab/feature.py cache_to_vw --features=caffe_fc7 --dataset=flickr

x Compute fc7 features on wikipaintings
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc7 --dataset=wikipaintings --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc7 --dataset=wikipaintings && python vislab/feature.py cache_to_vw --features=caffe_fc7 --dataset=wikipaintings

x Compute fc7 features on pinterest
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc7 --dataset=pinterest_80k --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc7 --dataset=pinterest_80k && python vislab/feature.py cache_to_vw --features=caffe_fc7 --dataset=pinterest_80k

x Distribute features to all machines

x Compute fc7 on PASCAL
    cd ~/work/vislab-git && python vislab/feature.py compute --features=caffe_fc7 --dataset=pascal --num_workers=20 && python vislab/feature.py cache_to_h5 --features=caffe_fc7 --dataset=pascal && python vislab/feature.py cache_to_vw --features=caffe_fc7 --dataset=pascal

- Compute mc_bit features on flickr
    cd ~/work/vislab-git && python vislab/feature.py compute --features=mc_bit --dataset=flickr --num_workers=20 && python vislab/feature.py cache_to_h5 --features=mc_bit --dataset=flickr && python vislab/feature.py cache_to_vw --features=mc_bit --dataset=flickr

o Compute mc_bit features on pinterest
    cd ~/work/vislab-git && python vislab/feature.py compute --features=mc_bit --dataset=pinterest_80k --num_workers=20 && python vislab/feature.py cache_to_h5 --features=mc_bit --dataset=pinterest_80k && python vislab/feature.py cache_to_vw --features=mc_bit --dataset=pinterest_80k

- Dig up old stuff:
    : not much to be found!
    - found predict databases dump from oct31
    - no features, no trained classifiers (so have to re-train pascal)

- Make list of prediction experiments that need to be run
    - always use 8 workers

## Tomorrow

- restart data view server

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
x try out predict command on cluster
