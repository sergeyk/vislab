"""
ImageNet detection challenge.

Everything loaded from files, and images distributed with dataset.
"""
from vislab.datasets.pascal import load_annotation_files
import vislab
import os
import pandas as pd
import glob


def load_imagenet(year='2013', force=False, args=None):
    """
    TODO: currently only loads val split.
    TODO: current hard-coded to be 2013 split.

    Load all the annotations, including object bounding boxes.
    Loads XML data in args['num_workers'] threads using joblib.Parallel.

    Warning: this takes a few minutes to load from scratch!
    """
    if args is None:
        args = {'num_workers': 8}

    cache_filename = \
        vislab.config['paths']['shared_data'] + \
        '/ilsvrc{}_dfs.h5'.format(year)
    if not force and os.path.exists(cache_filename):
        images_df = pd.read_hdf(cache_filename, 'images_df')
        objects_df = pd.read_hdf(cache_filename, 'objects_df')
        return images_df, objects_df

    # Load all annotation file data (should take < 30 s).
    # TODO: concat the dataframes here
    splits = ['val']
    for split in splits:
        annotation_filenames = glob.glob('{}/DET_bbox_{}/*.xml'.format(
            vislab.config['paths']['ILSVRC{}'.format(year)], split))
        images_df, objects_df = load_annotation_files(
            annotation_filenames, args['num_workers'])
        images_df['_split'] = split

    # Make sure that all labels are either True or False.
    images_df = images_df.fillna(False)

    # Propagate split info to objects_df
    objects_df['split'] = np.repeat(
        images_df['_split'].values, images_df['_num_objects'].values)

    images_df.to_hdf(cache_filename, 'images_df', mode='w')
    objects_df.to_hdf(cache_filename, 'objects_df', mode='a')
    return images_df, objects_df
