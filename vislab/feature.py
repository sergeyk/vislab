"""
Copyright Sergey Karayev - 2013.
Written during internship at Adobe CTL, San Francisco.

Every feature imported in features/__init__.py must take
image_filenames and image_ids, and returns good_image_ids and feats,
where good_image_ids may be a subset of image_ids.
"""
import os
import sys
import functools
import pandas as pd
import numpy as np
import cPickle
import bson
import vislab.utils.cmdline
import vislab.dataset
import vislab.features
import vislab.vw3

DB_NAME = 'vislab_feats'

FEATURES = {
    # does not look at image at all:
    'noise': {
        'cpus_per_task': 1, 'mem': 1000, 'chunk_size': -1, 'fn': None
    },

    # python:
    'size': {
        'fn': vislab.features.size,
        'cpus_per_task': 1, 'mem': 1000, 'chunk_size': -1
    },

    'gist_256': {
        'fn': functools.partial(vislab.features.gist, max_size=256),
        'cpus_per_task': 2, 'mem': 2000, 'chunk_size': -1
    },

    # caffe:
    'caffe_imagenet': {
        'fn': functools.partial(vislab.features.caffe, layer='prob'),
        'cpus_per_task': 4, 'mem': 3000, 'chunk_size': 30,
    },

    'caffe_fc6': {
        'fn': functools.partial(vislab.features.caffe, layer='fc6'),
        'cpus_per_task': 4, 'mem': 3000, 'chunk_size': 30,
    },

    'caffe_fc7': {
        'fn': functools.partial(vislab.features.caffe, layer='fc7'),
        'cpus_per_task': 4, 'mem': 3000, 'chunk_size': 30,
    },

    # matlab:
    'dsift_llc_1000': {
        'fn': vislab.features.dsift_llc,
        'cpus_per_task': 3, 'mem': 3000, 'chunk_size': 20
    },

    'lab_hist': {
        'fn': vislab.features.lab_hist,
        'cpus_per_task': 4, 'mem': 3000, 'chunk_size': 30
    },

    'mc_bit': {
        'fn': vislab.features.mc_bit,
        'cpus_per_task': 1, 'mem': 7000, 'chunk_size': 10
    },

    'gbvs_saliency': {
        'fn': vislab.features.gbvs_saliency,
        'cpus_per_task': 4, 'mem': 3000, 'chunk_size': 10
    },
}


def extract_features(
        dataset_df, dataset_name, feat_name, force=False,
        mem=3000, cpus_per_task=2, num_workers=1):
    """
    Extract features for each image in a list of image ids.
    Only those images that do not already have feature information are
    processed.
    Features are stored in a collection in the features mongo database
    as they are computed.

    Parameters
    ----------
    dataset_df: pandas.DataFrame
    dataset_name: string
    feat_name: string
        Must be in KNOWN_FEATURES.
    force: boolean [False]
        Compute feature even if it is already in database.
    mem: int [3000]
    cpus_per_task: int [2]
    num_workers: int [1]

    Returns
    -------
    features: ndarray
    """
    # Check that the settings are valid.
    assert(feat_name in FEATURES and
           cpus_per_task >= FEATURES[feat_name]['cpus_per_task'] and
           mem >= FEATURES[feat_name]['mem'])

    # Determine the cache filename, thereby creating the right directory
    dirname = vislab.util.makedirs('{}/{}'.format(
        vislab.config['paths']['feats'], dataset_name))
    h5_filename = '{}/{}.h5'.format(dirname, feat_name)

    collection = _get_feat_collection(dataset_name, feat_name)
    collection.ensure_index('image_id')

    # Exclude ids that already have computed features in the database.
    image_ids = dataset_df.index.tolist()
    if not force:
        computed_image_ids = [
            x['image_id'] for x in collection.find(fields=['image_id'])]
        num_ids = len(image_ids)
        image_ids = list(set(image_ids) - set(computed_image_ids))
        print("Cut down on {} existing out of {} total image ids.".format(
            num_ids - len(image_ids), num_ids))
        if len(image_ids) < 1:
            return

    # Features that are computed without any input are a special case.
    if feat_name == 'noise':
        X = np.random.rand(dataset_df.shape[0], 2).astype('float32')
        df = pd.DataFrame(
            data={'feat': [row for row in X]},
            index=dataset_df.index)

        df.to_hdf(h5_filename, 'df', mode='w')
        return

    # Next, everything else.
    num_chunks = max(1, len(image_ids) / FEATURES[feat_name]['chunk_size'])
    id_chunks = np.array_split(image_ids, num_chunks)
    args_list = [
        (id_chunk.tolist(), feat_name, dataset_name)
        for id_chunk in id_chunks
    ]
    vislab.utils.distributed.map_through_rq(
        vislab.feature._extract_features_for_image_ids, args_list,
        dataset_name + '_' + feat_name,
        num_workers=num_workers, mem=mem, cpus_per_task=cpus_per_task,
        async=(num_workers > 1))


def _extract_features_for_image_ids(
        image_ids, feat_name, dataset_name):
    """
    Download images, compute features, and store to database, for the
    given list of image_ids in dataset_name, with feat_name.
    """
    collection = _get_feat_collection(dataset_name, feat_name)
    image_filenames = vislab.dataset.fetch_image_filenames_for_ids(
        image_ids, dataset_name)
    if len(image_ids) == 0:
        print("Could not load any images from {}".format(image_ids))
        return
    image_ids, feats = FEATURES[feat_name]['fn'](image_ids, image_filenames)
    _store_in_db(collection, image_ids, feats)


def _get_feat_collection(dataset_name, feat_name):
    db_name = '{}_{}'.format(DB_NAME, dataset_name)
    return vislab.util.get_mongodb_client()[db_name][feat_name]


def _store_in_db(collection, image_ids, feats):
    """
    Store the given features for the given ids in the database.

    Parameters
    ----------
    collection: pymongo.Collection
    image_ids: list of string
    feats: list of ndarray
    """
    for image_id, feat in zip(image_ids, feats):
        collection.update({'image_id': image_id}, {
            'image_id': image_id,
            'feat': bson.Binary(cPickle.dumps(feat, protocol=2))
        }, upsert=True)


def _cache_to_h5(
        dataset_name, image_ids, feat_name, standardize=False, force=False):
    print force
    dirname = vislab.util.makedirs(
        os.path.join(vislab.config['paths']['feats'], dataset_name))

    filename = '{}/{}.h5'.format(dirname, feat_name)
    if not force and os.path.exists(filename):
        print("Cached file for {}: {} already exists.".format(
            dataset_name, feat_name))
        return

    collection = _get_feat_collection(dataset_name, feat_name)
    print('{} records in collection'.format(collection.count()))

    cursor = collection.find({'image_id': {'$in': image_ids}})
    print("{} records in cursor".format(cursor.count()))

    image_ids_ = []
    feats = []
    for document in cursor:
        image_ids_.append(str(document['image_id']))
        feat = cPickle.loads(document['feat'])
        if feat.dtype in [float, 'float32', 'float64']:
            feat = feat.astype('float32')
        feats.append(feat)

    # Drop all feature vectors with NaN's
    df = pd.DataFrame(np.vstack(feats), image_ids_)
    df = df.dropna()

    # drop duplicates
    df['image_id'] = df.index
    df = df.drop_duplicates('image_id')
    del df['image_id']
    print("{} rows remain after duplicates removed".format(df.shape[0]))

    if standardize and feat.dtype == 'float32':
        print("Standardizing data")
        df.values = (df.values - df.values.mean(0)) / df.values.std(0)

    df.to_hdf(filename, 'df', complib='blosc')


def _cache_to_vw(
        dataset_name, image_ids, feature_name, standardize=None, force=False):
    """
    Output VW-formatted feature values to GZIP'd file for the given image ids.
    If h5 cache of features exists, writes from there.
    If not, writes directly from database.
    """
    assert(feature_name in FEATURES)
    dirname = os.path.join(vislab.config['paths']['feats'], dataset_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = '{}/{}.txt.gz'.format(dirname, feature_name)
    if not force and os.path.exists(filename):
        print("Cached file already exists.")
        return

    h5_filename = '{}/{}/{}.h5'.format(
        vislab.config['paths']['feats'], dataset_name, feature_name)

    # If HDF5 cache exists, write from it, shuffling lines.
    if os.path.exists(h5_filename):
        _write_features_for_vw_from_h5(
            image_ids, dataset_name, feature_name, filename)

    # Otherwise, write from the database.
    else:
        args = (image_ids, dataset_name, feature_name)
        func = _write_features_for_vw_from_db

        print("Outputting to gzipped feature file {}".format(filename))
        python_cmd = 'python -c "{}"'.format(
            vislab.util.pickle_function_call(func, args))
        cmd = "{} | gzip > {}".format(python_cmd, filename)
        print(cmd)
        vislab.util.run_through_bash_script([cmd])


def _write_features_for_vw_from_h5(
        image_ids, dataset_name, feature_name, output_filename):
    """
    Does not standardize features, assumes already standardized.
    """
    h5_filename = '{}/{}/{}.h5'.format(
        vislab.config['paths']['feats'], dataset_name, feature_name)
    df = pd.read_hdf(h5_filename, 'df')
    df.index = df.index.astype(str)
    sys.stderr.write(
        "_write_features_for_vw_from_h5: Count for feature {}: {}\n".format(
            feature_name, df.shape[0]))

    # NOTE: this line is necessary... saw segfaults without it
    good_ids = [x for x in image_ids if x in df.index]
    df = df.ix[good_ids]
    df = df.dropna()

    # Shuffle!
    df = df.iloc[np.random.permutation(df.shape[0])]

    vislab.vw3.write_data_in_vw_format(df, feature_name, output_filename)


def _write_features_for_vw_from_db(image_ids, dataset_name, feature_name):
    # Connect to DB and get a cursor with all image_ids.
    collection = _get_feat_collection(dataset_name, feature_name)
    cursor = collection.find({'image_id': {'$in': image_ids}})
    fn_name = '_write_features_for_vw_from_db'
    sys.stderr.write("{}: Count for feature {}: {}. Matching ids: {}\n".format(
        fn_name, feature_name, collection.count(), cursor.count()))

    # Output VW-formatted features:
    # id |feature_name ind:val ind:val ind:val ...
    for document in cursor:
        feat = cPickle.loads(document['feat'])
        s = vislab.vw3._feat_for_vw(document['image_id'], feature_name, feat)
        sys.stdout.write(s + '\n')


## Command-line interface.
def compute(args=None):
    """
    Extract features of the requested type for all images in AVA.
    """
    if args is None:
        args = vislab.utils.cmdline.get_args(
            'feature', 'compute', ['dataset', 'processing', 'feature'])
    df = vislab.dataset.get_df_with_args(args)

    for feature in args.features:
        extract_features(
            df, args.dataset, feature, args.force_features,
            args.mem, args.cpus_per_task, args.num_workers)


def cache_to_h5(args=None):
    """
    Output features in the database for the ids in the loaded dataset to
    HDF5 cache file, one for each type of feature.
    """
    _cache(_cache_to_h5, 'h5', args)


def cache_to_vw(args=None):
    """
    Output features in the database for the ids in the loaded dataset to
    VW format gzip file, one for each type of feature.
    """
    _cache(_cache_to_vw, 'vw', args)


def _cache(fn, name, args=None):
    if args is None:
        args = vislab.utils.cmdline.get_args(
            'feature', 'cache_to_{}'.format(name),
            ['dataset', 'processing', 'feature']
        )
    df = vislab.dataset.get_df_with_args(args)
    image_ids = df.index.tolist()
    for feature in args.features:
        fn(args.dataset, image_ids, feature,
           args.standardize, args.force_features)


if __name__ == '__main__':
    possible_functions = {
        'compute': compute,
        'cache_to_h5': cache_to_h5,
        'cache_to_vw': cache_to_vw
    }
    vislab.utils.cmdline.run_function_in_file(__file__, possible_functions)
