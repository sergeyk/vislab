"""
Functions to compute various image features.
Some features are implemented in their own files.

Some functions take a list of image_ids, some a single image_id.
The _extract_features_for_image_ids function deals with that.

Features must be stored as numpy arrays.

Copyright Sergey Karayev - 2013.
Written during internship at Adobe CTL, San Francisco.
"""
import numpy as np
import os
import sys
import pandas
import pandas as pd
import bson
import cPickle
import vislab
import vislab.vw3


DB_NAME = 'ava_feat_db'

FEAT_DIRNAME = vislab.util.makedirs(
    vislab.repo_dirname + '/data/feats')

KNOWN_FEATURES = [
    'noise',
    'size',
    'gist_256',  # python
    'dsift_llc_1000',  # matlab
    'color_K150_s1.00_3x3_1x1',  # python
    'lab_hist',  # matlab
    'mc_bit',  # matlab
    'gbvs_saliency',  # matlab
    'decaf_fc6_flatten',  # decaf: python
    'decaf_fc6',  # decaf: python
    'decaf_tuned_fc6_flatten',  # decaf: python, tuned on style data
    'decaf_tuned_fc6_flatten_ud',  # decaf: python, tuned on style data
    'decaf_tuned_fc6',  # decaf: python, tuned on style data
    'decaf_tuned_fc6_ud',  # decaf: python, tuned on style data
    'decaf_imagenet',  # decaf: python
]

RECOMMENDED_SETTINGS = {
    'gbvs_saliency': {'cpus_per_task': 4, 'mem': 3000},
    'lab_hist': {'cpus_per_task': 4, 'mem': 3000},
    'mc_bit': {'cpus_per_task': 1, 'mem': 7000},
    'color_K150_s1.00_3x3_1x1': {'cpus_per_task': 2, 'mem': 3000},
    'dsift_llc_1000': {'cpus_per_task': 3, 'mem': 3000},
    'gist_256': {'cpus_per_task': 2, 'mem': 2000},
    'decaf_fc6_flatten': {'cpus_per_task': 4, 'mem': 2500},
    'decaf_fc6': {'cpus_per_task': 4, 'mem': 2500},
    'decaf_fc6_tuned_flatten': {'cpus_per_task': 4, 'mem': 3500},
    'decaf_fc6_tuned_flatten_ud': {'cpus_per_task': 4, 'mem': 3500},
    'decaf_tuned_fc6': {'cpus_per_task': 4, 'mem': 3500},
    'decaf_tuned_fc6_ud': {'cpus_per_task': 4, 'mem': 3500},
    'decaf_imagenet': {'cpus_per_task': 4, 'mem': 2500}
}


def _extract_features_for_image_ids(
        image_ids, feature_name, dataset_name, force):
    """
    Function to process a short list of image_ids with a feature function, and
    to store the results in the database.
    """
    # Determine filenames of the images (download images if not cached).
    image_filenames = [
        vislab.image.fetch_image_filename_for_id(image_id, dataset_name)
        for image_id in image_ids
    ]

    # Drop images that we couldn't get a filename for.
    image_filenames = [x for x in image_filenames if x is not None]
    image_ids = [
        y for x, y in zip(image_filenames, image_ids) if x is not None
    ]

    if len(image_ids) == 0:
        print("Dropped all images!")
        return

    # Connect to the feature database.
    collection = get_feat_collection(feature_name)
    collection.ensure_index('image_id')

    # Compute the features.
    # Some feature functions process groups of ids, and some process them
    # one by one.
    if feature_name in ['dsift_llc_1000']:
        image_ids, feats = vislab.features.dsift.dsift_llc(
            image_filenames, image_ids)

    elif feature_name in ['color_K150_s1.00_3x3_1x1']:
        feats = vislab.features.color_histogram(
            image_filenames, image_ids,
            K=150, sp_grid=[(3, 3), (1, 1)], sigma=1)

    elif feature_name in ['mc_bit']:
        image_ids, feats = vislab.features.mc_bit(
            image_filenames, image_ids)

    elif feature_name == 'gbvs_saliency':
        feats = vislab.features.gbvs_saliency(image_filenames, image_ids)

    elif feature_name in ['lab_hist']:
        feats = vislab.features.lab_hist(image_filenames, image_ids)

    elif feature_name in ['size']:
        feats = [
            vislab.features.size(filename, id_)
            for filename, id_ in zip(image_filenames, image_ids)
        ]

    elif feature_name in ['gist_256']:
        feats = [
            vislab.features.gist(filename, id_, 256)
            for filename, id_ in zip(image_filenames, image_ids)
        ]

    elif feature_name == 'decaf_imagenet':
        image_ids, feats = vislab.features.decaf_feat(
            image_filenames, image_ids, 'imagenet', tuned=False)

    elif feature_name in ['decaf_fc6']:
        image_ids, feats = vislab.features.decaf_feat(
            image_filenames, image_ids, 'fc6_cudanet_out', tuned=False)

    elif feature_name in ['decaf_fc6_flatten']:
        image_ids, feats = vislab.features.decaf_feat(
            image_filenames, image_ids, '_decaf_fc6_flatten_out', tuned=False)

    elif feature_name in ['decaf_tuned_fc6', 'decaf_tuned_fc6_ud']:
        image_ids, feats = vislab.features.decaf_feat(
            image_filenames, image_ids, 'fc6_cudanet_out', tuned=True)

    elif feature_name in ['decaf_tuned_fc6_flatten', 'decaf_tuned_fc6_flatten_ud']:
        image_ids, feats = vislab.features.decaf_feat(
            image_filenames, image_ids, '_decaf_fc6_flatten_out', tuned=True)

    else:
        raise("This feature_name is not implemented!")

    _store_in_db(collection, image_ids, feats)


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


def extract_features(
        dataset_df, dataset_name, feature_name, force=False,
        mem=3000, cpus_per_task=2, num_workers=1):
    """
    Extract features for each image in a list of image ids.
    Features are stored in a collection in the features mongo database
    as they are computed.

    Parameters
    ----------
    dataset_df: pandas.DataFrame
    dataset_name: string
    feature_name: string
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
    assert(feature_name in KNOWN_FEATURES)
    if feature_name in RECOMMENDED_SETTINGS:
        assert(cpus_per_task >=
               RECOMMENDED_SETTINGS[feature_name]['cpus_per_task'])
        assert(mem >= RECOMMENDED_SETTINGS[feature_name]['mem'])

    # Determine the cache filename, thereby creating the right directory
    dirname = vislab.util.makedirs('{}/{}'.format(FEAT_DIRNAME, dataset_name))
    h5_filename = '{}/{}.h5'.format(dirname, feature_name)

    collection = get_feat_collection(feature_name)

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

    ## First, consider features that are computed without any input.
    if feature_name == 'noise':
        X = np.random.rand(dataset_df.shape[0], 2).astype('float32')
        df = pandas.DataFrame(
            data={'feat': [row for row in X]},
            index=dataset_df.index)

        df.to_hdf(h5_filename, 'df', mode='w')
        return

    elif feature_name in ['size', 'gist_256']:
        args_list = [
            ([id_, ], feature_name, dataset_name, force)
            for id_ in image_ids
        ]

    elif feature_name in [
            'dsift_llc_1000', 'color_K150_s1.00_3x3_1x1',
            'mc_bit', 'lab_hist', 'gbvs_saliency',
            'decaf_imagenet', 'decaf_fc6', 'decaf_fc6_flatten',
            'decaf_tuned_fc6', 'decaf_tuned_fc6_flatten',
            'decaf_tuned_fc6_ud', 'decaf_tuned_fc6_flatten_ud']:
        chunk_sizes = {
            'decaf_fc6': 60,
            'decaf_fc6_flatten': 60,
            'decaf_tuned_fc6': 60,
            'decaf_tuned_fc6_flatten': 60,
            'decaf_tuned_fc6_ud': 60,
            'decaf_tuned_fc6_flatten_ud': 60,
            'decaf_imagenet': 60,
            'dsift_llc_1000': 20,
            'color_K150_s1.00_3x3_1x1': 40,
            'mc_bit': 10,
            'lab_hist': 30,
            'gbvs_saliency': 10
        }
        num_chunks = len(image_ids) / chunk_sizes[feature_name]
        id_chunks = np.array_split(image_ids, num_chunks)
        args_list = [
            (id_chunk.tolist(), feature_name, dataset_name, force)
            for id_chunk in id_chunks
        ]

    else:
        raise Exception('Unimplemented')

    vislab.utils.distributed.map_through_rq(
        _extract_features_for_image_ids, args_list,
        dataset_name + '_' + feature_name,
        num_workers=num_workers, mem=mem, cpus_per_task=cpus_per_task,
        async=(num_workers > 1))


def get_feat_collection(feature_name):
    return vislab.util.get_mongodb_client()[DB_NAME][feature_name]


def _cache_to_h5(dataset_name, image_ids, feat_name, force=False):
    dirname = os.path.join(FEAT_DIRNAME, dataset_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = '{}/{}.h5'.format(dirname, feat_name)
    if not force and os.path.exists(filename):
        print("Cached file already exists.")
        return

    collection = get_feat_collection(feat_name)
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
    X = df.values
    index = df.index

    # Standardize data if needed. But don't standardize if:
    #   - rows are L1-normalized
    #   - value range is within [-1, 1]
    l1_normalized = np.allclose(X.sum(1), 0, atol=1e-5)
    good_range = X.min() >= -1 and X.max() <= 1
    if feat.dtype == 'float32' and not (l1_normalized or good_range):
        print("Standardizing data")
        mean = X.mean(0)
        std = X.std(0)
        X -= mean
        X /= std

    df = pandas.DataFrame({'feat': [row for row in X]}, index)

    # drop duplicates
    df['image_id'] = df.index
    df = df.drop_duplicates('image_id')
    del df['image_id']

    try:
        df.to_hdf(filename, 'df', mode='w')
    except:
        df.to_pickle(filename)


def _cache_to_vw(dataset_name, image_ids, feature_name, force=False):
    """
    Output VW-formatted feature values to GZIP'd file for the given image ids.
    If h5 cache of features exists, writes from there.
    If not, writes directly from database.
    """
    assert(feature_name in KNOWN_FEATURES)
    dirname = os.path.join(FEAT_DIRNAME, dataset_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = '{}/{}.txt.gz'.format(dirname, feature_name)
    if not force and os.path.exists(filename):
        print("Cached file already exists.")
        return

    h5_filename = '{}/{}/{}.h5'.format(
        FEAT_DIRNAME, dataset_name, feature_name)

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
        FEAT_DIRNAME, dataset_name, feature_name)
    try:
        df = pd.read_hdf(h5_filename, 'df')
    except:
        df = pd.read_pickle(h5_filename)
    sys.stderr.write(
        "_write_features_for_vw_from_h5: Count for feature {}: {}\n".format(
            feature_name, df.shape[0]))
    df.index = df.index.astype(str)

    # TODO: why does this segfault without this line? stupid
    good_ids = [x for x in image_ids if x in df.index]
    df = df.ix[good_ids]
    df = df.dropna()

    # Shuffle!
    df = df.iloc[np.random.permutation(df.shape[0])]

    vislab.vw3.write_data_in_vw_format(df, feature_name, output_filename)


def _write_features_for_vw_from_db(image_ids, dataset_name, feature_name):
    # Connect to DB and get a cursor with all image_ids.
    collection = get_feat_collection(feature_name)
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


def load_features_for_image_ids(image_ids, feature_name):
    """
    Load features from cache of database into a DataFrame.
    Note that not all image_ids may be present in the cache.
    The returned DataFrame has only those image_ids that are in the database.

    Parameters
    ----------
    image_ids: list of string
    feature_name: name of the feature function

    Returns
    -------
    df: pandas.DataFrame
    """
    assert(feature_name in KNOWN_FEATURES)
    filename = '{}/{}.h5'.format(FEAT_DIRNAME, feature_name)
    with pandas.get_store(filename) as f:
        df = f['df']
        df = df.ix[image_ids].dropna()
    return df
