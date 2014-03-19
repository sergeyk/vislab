"""
ImageNet classification and detection challenges.

Everything loaded from files, and images distributed with dataset.
"""
import os
import pandas as pd
import glob
import scipy.io
import networkx as nx
import numpy as np
import multiprocessing
import vislab
from vislab.datasets.pascal import load_annotation_files


class ImagenetGraph(object):
    """
    Represents the ImageNet structure, loaded from .mat files provided
    in the ILSVRC2013_devkit.

    Download devkit from [1] and untar into devkit_dirname.
    Then download meta_10k from [2] and place into devkit_dirname/data.

    [1]: http://imagenet.stanford.edu/image/ilsvrc2013/ILSVRC2013_devkit.tgz
    [2]: https://dl.dropboxusercontent.com/u/44891/research/meta_10k.mat
    """
    def __init__(self, metafile, type='1k'):
        """
        Parameters
        ----------
        type: string
            In ['1k', '10k', 'det'].
        """
        data = scipy.io.loadmat(metafile)['synsets']
        if not type == '10k':
            data = data[0]

        g = nx.DiGraph()

        # First pass: add nodes.
        wnids = []
        for node in data:
            if type == '10k':
                node = node[0]

            wnid = str(node[1][0])
            wnids.append(wnid)
            g.add_node(wnid, {'words': node[2][0]})

        # Second pass: add edges.
        for i, node in enumerate(data):
            if type == '10k':
                node = node[0]

            if type == 'det':
                children = node[4].flatten()
            else:
                children = node[5][0]

            # Children are IDs from the original metafile, which is 1-indexed.
            for child in children:
                g.add_edge(wnids[i], wnids[child - 1])

        self.g = g

    def node_name(self, wnid):
        word = self.g.node[wnid]['words'].split(',')[0]
        return '{} ({})'.format(word, wnid)

    def get_all_successors(self, wnid):
        children = self.g.successors(wnid)
        all_children = list(children)
        for child in children:
            all_children += self.get_all_successors(child)
        return all_children

    def get_leaf_nodes(self, wnids):
        return [
            wnid for wnid in wnids
            if not self.g.successors(wnid)
        ]


def load_imagenet_detection(year='2013', force=False, args=None):
    """
    TODO: currently only loads val split.
    TODO: current hard-coded to be 2013 split.

    Load all the annotations, including object bounding boxes.
    Loads XML data in args['num_workers'] threads using joblib.Parallel.

    Warning: this takes a few minutes to load from scratch!
    """
    if args is None:
        args = {'num_workers': multiprocessing.cpu_count()}

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
        images_df['_split'].values,
        images_df['_num_objects'].values.astype(int)
    )

    images_df.to_hdf(cache_filename, 'images_df', mode='w')
    objects_df.to_hdf(cache_filename, 'objects_df', mode='a')
    return images_df, objects_df
