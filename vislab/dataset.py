"""
Construct dicts containing identifying information and DataFrames
for train, val, test splits of data, for use with classifiers.

See docs/datasets.md for more information on Vislab datasets.
"""
import numpy as np
import pandas as pd
import requests
import os
import subprocess
import csv
import vislab
import vislab.utils.cmdline
import vislab.utils.distributed
import vislab.datasets


def download_images(df, dirname, num_cpus=6):
    """
    Download images from a dataset to a directory.

    Parameters
    ----------
    df: pandas.DataFrame
        Must have column 'image_url', and a unique index that can be
        used as filenames.
    dirname: str
        Images will be downloaded to dirname/originals.
    num_cpus: int
        Downloading will be parallelized with this many processes.
    """
    vislab.util.makedirs(dirname + '/original')
    df[['image_url']].to_csv(
        dirname + '/image_url.csv',
        sep=' ', header=None, quoting=csv.QUOTE_ALL
    )
    cmd = 'parallel --gnu -a image_url.csv --colsep=" " -j {} '.format(
        num_cpus)
    cmd += '"wget {{2}} -P original/{{1}}.jpg"\n'
    with open(dirname + '/download_cmd.sh', 'w') as f:
        f.write(cmd)
    subprocess.call(
        ['sh', dirname + '/download_cmd.sh'], shell=True, cwd=dirname)


def resize_images(dirname, size, num_cpus=6):
    vislab.util.makedirs(dirname + '/{}px'.format)
    # TODO clean up below
    [
        'mkdir {}px'.format(size),
        'parallel -a _temp_photo_names.txt -j {0} "convert original/{{}} -resize {1}\> {1}px/{{}}"'.format(num_cpus, size),
        'parallel -a _temp_photo_names.txt -j {} "identify original/{{}}" > _sizes.txt'.format(num_cpus),
    ]


def get_image_sizes(dirname, num_cpus=6):
    # TODO: borrow code from above, or write job-queue based approach
    pass


def dl_and_resize_images(df, dirname, size, num_cpus=6):
    """
    Download images from a dataset to a directory.

    Parameters
    ----------
    df: pandas.DataFrame
        Must have column 'image_url', and a unique index that can be
        used as filenames.
    dirname: str
        Images will be downloaded to dirname/originals.
    num_cpus: int
        Downloading will be parallelized with this many processes.
    """
    # Load in the image sizes
    size_df = pd.read_csv(dirname + '/_sizes.txt', sep=' ', header=None)
    size_df = pd.DataFrame({
        'photo_id': [_[:-4] for _ in size_df[0]],
        'width': [int(_.split('x')[0]) for _ in size_df[2]],
        'height': [int(_.split('x')[1]) for _ in size_df[2]]
    })
    return size_df


def get_train_test_split(df_, test_frac=0.2, random_seed=42):
    """
    Split DataFrame into train and test subsets, by preserving label
    ratios.
    """
    np.random.seed(random_seed)

    N = df_.shape[0]
    df_ = df_.iloc[np.random.permutation(N)]

    # Get equal amount of test_frac of each label
    counts = df_.sum(0).astype(int)
    min_count = int(round(counts[counts.argmin()] * test_frac))
    test_balanced_set = np.concatenate([
        df_.index[np.where(df_[l])[0][:min_count]]
        for l, count in counts.iteritems()
    ]).tolist()

    # Then add enough of the rest to get to test_frac of total.
    remaining_ind = df_.index.diff(test_balanced_set).tolist()
    np.random.shuffle(remaining_ind)
    num_test = int(round(N * test_frac))
    num_to_add = num_test - len(test_balanced_set)
    if num_to_add > 0:
        test_balanced_set += remaining_ind[:num_to_add]
    else:
        test_balanced_set = np.random.choice(
            test_balanced_set, num_test, replace=False)

    split = pd.Series('train', index=df_.index, name='_split')
    split.ix[test_balanced_set] = 'test'
    return split


def get_bool_df(df, column_name, min_positive_examples=-1):
    """
    Return a boolean DataFrame whose columns consist of unique
    values of df[column_name] that have more than the required
    number of positive examples.

    Parameters
    ----------
    df: pandas.DataFrame
    column_name: string
    min_positive_examples: int [-1]
        Only take those labels with more examples than this.
    """
    assert(column_name in df.columns)
    df = df.dropna(subset=[column_name])

    freqs = df[column_name].value_counts()

    # Filter out vals with less than min_pos examples.
    if min_positive_examples > 0:
        freqs = freqs[freqs >= min_positive_examples]
    vals = freqs.index.tolist()
    df = df[df[column_name].apply(lambda x: x in vals)]

    # Expand values into own boolean DataFrame.
    bool_df = pd.DataFrame(index=df.index)
    for val in vals:
        ascii_name = val.replace(' ', '_').encode('ascii', 'ignore')
        if len(column_name) > 0:
            ascii_name = column_name + '_' + ascii_name
        bool_df[ascii_name] = (df[column_name] == val)
    return bool_df


def subsample_dataset(df, num_images=-1, random_seed=42):
    """
    Return a subsampled version of the dataset, with num_images images.
    Take images randomly, according to random_seed.

    Note: Does NOT permute images if df is of size num_images.
    """
    if num_images < 0 or num_images >= df.shape[0]:
        return df
    np.random.seed(random_seed)
    ind = np.random.permutation(df.shape[0])[:num_images]
    return df.iloc[ind]


def fetch_image_filenames_for_ids(image_ids, dataset_name):
    """
    Return list of image filenames for given image_ids in dataset_name.
    If the images are not already present on disk, downloads them to
    cache location.

    Parameters
    ----------
    image_ids: list of string
    dataset_name: string

    Returns
    -------
    good_filenames: list of string
        Only filenames of images that actually exist on disk.
    """
    image_dirname = vislab.util.makedirs(os.path.join(
        vislab.config['paths']['images'], dataset_name))

    df = load_dataset_df(dataset_name)

    if 'image_filename' in df.columns:
        filenames = df['image_filename'].loc[image_ids]
    else:
        assert 'image_url' in df.columns
        filenames = [
            os.path.join(image_dirname, '{}.jpg'.format(image_id))
            for image_id in image_ids
        ]
        urls = df['image_url'].loc[image_ids]

    good_filenames = []
    for i in range(len(filenames)):
        filename = filenames[i]
        if os.path.exists(filename):
            good_filenames.append(filename)
            continue

        try:
            print("Download image for {}: {}".format(dataset_name, filename))
            r = requests.get(urls[i])
            with open(filename, 'wb') as f:
                f.write(r.content)
            good_filenames.append(filename)

        except Exception as e:
            print("Exception: {}".format(e))

    return good_filenames


def load_dataset_df(dataset_name, force=False):
    if dataset_name not in vislab.datasets.DATASETS:
        raise Exception('Unknown dataset.')
    return vislab.datasets.DATASETS[dataset_name]['fn'](force=force)


def get_df_with_args(args=None):
    """
    Use the parsed command line arguments to load the correct dataset.
    Assumes the relevant datasets have already been fetched.

    If this is not true, refer to the individual dataset code on info
    about how to load.
    """
    # Parse arguments.
    if args is None:
        args = vislab.utils.cmdline.get_args(
            'dataset', 'get_df', ['dataset', 'processing'])
    df = load_dataset_df(args.dataset, args.force)
    df = subsample_dataset(df, args.num_images, args.random_seed)
    return df


if __name__ == '__main__':
    possible_functions = {
        'get_df': get_df_with_args,
    }
    vislab.utils.cmdline.run_function_in_file(__file__, possible_functions)
