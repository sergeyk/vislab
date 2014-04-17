---
layout: default
title: Tutorial
---
# Tutorial

## Setting up

VisLab has the following external dependencies:

- [MongoDB][]: key-value store used for storing features
- [Redis][]: key-value store used for our job queue
- [Vowpal Wabbit][]: machine learning framework
- and a bunch of Python packages: we recommend using the [Anaconda distribution][], which provides most of them out of the box.
- some Matlab code for feature computation is required if you're not satisfied with the convolutional network feature.

<!-- - [Caffe] deep-learning framework (another Berkeley project)
    Caffe itself has many dependencies, such as OpenCV.
 -->
Additionally, you'll benefit from downloading our pre-packaged datasets.
Follow along for the link.

## Installing dependencies

We're going to describe the setup on OS X (tested with versions 10.8 and above).

[Homebrew][] is by far the best way to manage packages on OS X, so we assume that it is used.
On Linux, most `brew install` commands can be replaced with `sudo apt-get install`.

We also assume that the [Anaconda distribution] of Python is used.
If not, no problem: simply replace `conda install` with `pip install` -- but you will need to install *many* additional packages, such as 'numpy', yourself.

First, external dependencies.

    brew install mongo
    brew install redis
    brew install fftw
    brew install parallel
    brew install boost

Now the Python package dependencies.

    conda install pymongo
    conda install pyleargist
    conda install joblib
    conda install rq
    conda install husl

A little trick is needed for openmpi:

    conda remove openmpi
    brew install openmpi
    conda install mpi4py

Clone and install Vowpal Wabbit.

    git clone git@github.com:JohnLangford/vowpal_wabbit.git
    cd vowpal_wabbit
    make
    make install

<!-- To install [Caffe][], please follow the [instructions](http://caffe.berkeleyvision.org/installation.html). -->

### Getting vislab code and wikipaintings dataset

You'll need to pick a work directory that will contain the repository.
I use `~/work`, so this repository goes into `~/work/vislab-git`.

    mkdir ~/work
    cd ~/work
    git clone https://github.com/sergeyk/vislab.git vislab-git

Set up the Python path to contain this directory:

    export PYTHONPATH=$HOME/work/vislab-git:$PYTHONPATH

*All following commmands assume that you are performing them in this directory.*

[MongoDB]: https://www.mongodb.org/
[Redis]: http://redis.io/
[Vowpal Wabbit]: https://github.com/JohnLangford/vowpal_wabbit
[Caffe]: http://caffe.berkeleyvision.org
[Anaconda distribution]: https://store.continuum.io/cshop/anaconda/
[Homebrew]: http://brew.sh/

## Simple experiment

Now let's run a classification experiment on the Wikipaintings dataset.

#### Assembling the dataset

First order of business is assembling the dataset.
Originally, this involved scraping ~100K records from the Wikipaintings.org website.
No one should have to do that more than once, so simply download our pre-packaged data from this [Dropbox folder](https://www.dropbox.com/sh/8x6fdlftt2w3z4g/gM0Fx_4pLn).
Copy the whole thing to `vislab-git/data/shared` -- or simply symlink directly to the dropbox directory:

    mkdir data
    ln -s ~/Dropbox/vislab_data_shared data/shared

Next, copy over the default configuration.

    cp vislab/config.json.example vislab/config.json

Edit `config.json` to point to the folders for your local installation (the defaults should work if you've followed the instructions so far).

To use a lot of the functionality of the dataset, we need to start a MongoDB server.
In another shell, run

    ./scripts/start_mongo.sh

This will launch a server on port `27666`, which is canonically expected by the rest of the code.
You can ignore any errors about `numactl` being missing; don't worry, that only plays a role for our cluster setup.
MongoDB is used for storing image information, features, and classification results.

To check that you can load the dataset, open another terminal and fire up IPython Notebook:

    ipython notebook --pylab=inline notebooks

This should open a browser, where you can select the 'wikipaintings dataset' notebook.
Run the cells in it to make sure that the data quickly loads and works as expected.
If python complains about being unable to import packages, `conda install` or `pip install` them as needed.

Note: if the cell with `df = wikipaintings.get_basic_df()` doesn't complete within a few seconds, then the data wasn't found; check that you have correctly set the paths in `config.json`.

### Computing features on a dataset

The first step to classifying the dataset is computing some features on the images.
The dataset does not contain images, they will be downloaded as needed by the feature computation code.

For this demo, we won't actually look at the images, but simply use random noise as the only feature.

Computing features is an inherently parallel task, and we are going to use a Redis job queue and a pool of workers.
So, in another shell:

    ./scripts/start_redis.sh

Each chunk of jobs is executed by a client that downloads the images from their original URI's, processes them, and stores the computed features to database.

    python vislab/feature.py compute --dataset=wikipaintings --features=noise --num_workers=6 --cpus_per_task=4

After all the images have been processed, we write the data out to an HDF5 file for easier sharing and loading, and to a file that Vowpal Rabbit (our predictor) can read:

    python vislab/feature.py cache_to_h5 --dataset=wikipaintings --features=noise
    python vislab/feature.py cache_to_vw --dataset=wikipaintings --features=noise

You can always run `python vislab/feature.py compute -h` to read about the options.

### Training and testing prediction

TODO: talk about potenital pitfall: `gzcat` vs `zcat` on OS X.

The prediction task takes the feature to use as a flag, the MongoDB collection to store the results into, and the label to run on.

    python vislab/predict.py predict --dataset=wikipaintings --features=noise --collection_name=demo --prediction_label="style_*" --num_workers=6

This trains and tests on the Wikipaintings data, cross-validating parameters, and storing predictors to config-specified directory.

### Visualizing performance

To visualize the prediction performance, we use a notebook.
Take a look at the `wikipaintings results` notebook for an example.
Running through it will generate publiction-ready figures and tables.

### Applying predictors to a new dataset

We can take the predictors learned on the Wikipaintings dataset and apply them to another dataset.
In this demo, we will just run trained predictors on the same dataset:

    python vislab/predict.py predict --dataset=wikipaintings --source_dataset=wikipaintings --features=noise --collection_name=demo_predict_only --prediction_label="style_*" --num_workers=6

This is how predictors trained on one dataset can be applied to another dataset as "features."

### Hooking up a new dataset

If you get your data into a DataFrame format that the rest of the code expects, that's all that's really needed.

The required columns are 'image_id' (must be unique), 'image_url', and a column of whatever labels you care about.
After this, add a couple of lines to `get_df_with_args()` in `dataset.py` as needed, and add another couple of lines to `fetch_image_filename_for_id` in `aphrodite/image.py` as needed.

For an example of a scraped dataset, take a look at [vislab/datasets/wikipaintings.py](https://github.com/sergeyk/vislab/blob/master/vislab/datasets/wikipaintings.py), or [aphrodite/flickr.py](https://github.com/sergeyk/aphrodite/blob/master/aphrodite/flickr.py).

### Exploring data and prediction results in a UI

Coming soon.
