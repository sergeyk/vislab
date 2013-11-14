# VisLab

## Requirements

External dependencies:

- MongoDB


## Quick start.

Let's run a classification experiment on the Wikipaintings dataset, using style labels.


### Getting vislab code and wikipaintings dataset

You'll need to pick a work directory that will contain the repository folders.
I use `~/work`, so this repository goes into `~/work/vislab-git`.

    cd <work_directory>
    git clone https://github.com/sergeyk/vislab.git vislab-git
    cd vislab-git

Set up the Python path to contain this folder:

    export PYTHONPATH=$HOME/work/vislab:$PYTHONPATH

First order of business is assembling the dataset.
This involves scraping ~100K records from a website, but luckily I've already done that for you.
The data is in this [Dropbox folder](https://www.dropbox.com/sh/our2zcaaqfi2e6d/1rZs5J4xhl).
Copy it to `vislab/data/shared` -- I simply symlink directly to the dropbox folder.

Next, update configuration settings:

    cp vislab/config.json.example vislab/config.json

Then edit `config.json` to point to the folders for your local installation.

To use a lot of the functionality of the dataset, we need to start a MongoDB server.

    ./scripts/start_mongo.sh

This will launch a server on port `27666`, which is canonically expected by the rest of the code. You can ignore any errors about `numactl` being missing; that is only needed for cluster computation.
MongoDB is used for storing image information, features, and classification results.

To check that you can load the dataset, fire up IPython Notebook from the vislab directory:

    ipython notebook --pylab=inline notebooks

This should open a browser, where you can select the 'wikipaintings' notebook.
Run the cells in it to make sure that the data loads and works.
If python complains about being unable to import packages, `pip install` them as you see fit.
I recommend installing the [Anaconda distribution](https://store.continuum.io/cshop/anaconda/) that will have most of them already.

Note: if the cell with `df = wikipaintings.get_basic_dataset()` doesn't complete, then the data wasn't found; check that you have correctly set up the folders in `config.json` folders, as above.

