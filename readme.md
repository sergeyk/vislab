# VisLab

## Requirements

External dependencies:

- MongoDB
- Redis
- [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit)

For Python dependencies, I **highly** recommend installing the [Anaconda distribution](https://store.continuum.io/cshop/anaconda/) that will have most of the Python packages that we use.
You will also need a couple of Python packages that are not widely distributed:

- [aphrodite](https://github.com/sergeyk/aphrodite)
- [decaf](https://github.com/UCB-ICSI-Vision-Group/decaf-release)

## Quick start.

Let's run a classification experiment on the Wikipaintings dataset, using style labels.

### Getting vislab code and wikipaintings dataset

You'll need to pick a work directory that will contain the repository folders.
I use `~/work`, so this repository goes into `~/work/vislab-git`.

    cd <work_directory>
    git clone https://github.com/sergeyk/vislab.git vislab-git
    git clone https://github.com/sergeyk/aphrodite.git aphrodite-git

Set up the Python path to contain this folder:

    export PYTHONPATH=$HOME/work/vislab-git:$HOME/work/aphrodite-git:$PYTHONPATH

#### Assembling the dataset

First order of business is assembling the dataset.
This involves scraping ~100K records from a website, but luckily I've already done that for you.
The data is in this [Dropbox folder](https://www.dropbox.com/sh/our2zcaaqfi2e6d/1rZs5J4xhl).
Copy it to `vislab-git/data/shared` -- I simply symlink directly to the dropbox folder.

Next, update configuration settings:

    cd vislab-git
    cp vislab/config.json.example vislab/config.json

Then edit `config.json` to point to the folders for your local installation.

To use a lot of the functionality of the dataset, we need to start a MongoDB server.
In another shell, run

    ./scripts/start_mongo.sh

This will launch a server on port `27666`, which is canonically expected by the rest of the code.
You can ignore any errors about `numactl` being missing; that is only needed for cluster computation.
MongoDB is used for storing image information, features, and classification results.

To check that you can load the dataset, fire up IPython Notebook:

    ipython notebook --pylab=inline notebooks

This should open a browser, where you can select the 'wikipaintings dataset' notebook.
Run the cells in it to make sure that the data loads and works.
If python complains about being unable to import packages, `conda install` or `pip install` them as needed.

Note: if the cell with `df = wikipaintings.get_basic_dataset()` doesn't complete within a few seconds, then the data wasn't found; check that you have correctly set up the folders in `config.json` folders, as above.

### Computing features on a dataset

The first step to classifying the dataset is computing some features on the images.
The dataset does not contain images, they will be downloaded as needed by the feature computation code.

For this demo, we won't actually look at the images, but simply use random noise as our only feature.

We compute features in a distributed fashion, using a Redis job queue.
So, in another shell:

    ./scripts/start_redis.sh

Each chunk of jobs is executed by a client that downloads the images from their original URI's, then processes them, and stores the computed features to database.

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
