"""
Construct dicts containing identifying information and DataFrames
for train, val, test splits of data, for use with classifiers.

## Datasets

pascal:
    VOC2012, ~10K images tagged with 20 object classes. Multi-label.
ava:
    ~250K images with aesthetic ratings
ava_style:
    ~20K images from AVA that also have style labels
flickr:
    ~50K images with style labels
wikipaintings:
    ~100K images with style, genre, artist labels

## Format

- The main thing is for the DataFrame index to contain unique image ids,
and to be able to get the image url from the id.

- The classifier expects DataFrames with only two columns:
'label' and 'importance'.
- The 'label' column can contain
    - real values (regression)
    - -1/1 (binary classification)
    - or positive ints (multiclass classification).
"""
import numpy as np
import pandas as pd
import vislab
import vislab.utils.cmdline
import vislab.utils.distributed
import vislab.datasets


def get_train_test_split(df_, test_frac=0.2, random_seed=42):
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


def get_boolean_df(df, column_name, min_positive_examples=-1):
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
    np.random.seed(random_seed)
    if num_images < 0 or num_images >= df.shape[0]:
        return df
    ind = np.random.permutation(df.shape[0])[:num_images]
    return df.iloc[ind]


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

    # Load the dataset.
    if args.dataset == 'ava':
        df = vislab.datasets.ava.get_ava_df()

    elif args.dataset == 'ava_style':
        style_df = vislab.datasets.ava.get_style_df()
        ratings_df = vislab.datasets.ava.get_ratings_df()
        df = style_df.join(ratings_df)

    elif args.dataset == 'flickr':
        df = vislab.datasets.flickr.load_flickr_df(
            args.num_images, args.random_seed)

    elif args.dataset == 'wikipaintings':
        if args.label == 'artist':
            df = vislab.datasets.wikipaintings.get_artist_df()
        else:
            df = vislab.datasets.wikipaintings.get_style_df()

    elif args.dataset == 'pascal':
        df = vislab.datasets.pascal.get_clf_df()

    elif args.dataset == 'behance_photo':
        df = vislab.datasets.behance.get_photo_df()

    elif args.dataset == 'behance_illustration':
        df = pd.read_csv(
            vislab.repo_dirname + '/datasets/10k_illustrations_20_tags_3_images.csv',
            index_col=0)

    else:
        raise Exception('Unknown dataset.')

    # Subsample number of images if necessary.
    df = subsample_dataset(df, args.num_images, args.random_seed)

    return df


if __name__ == '__main__':
    possible_functions = {
        'get_df': get_df_with_args,
    }
    vislab.utils.cmdline.run_function_in_file(__file__, possible_functions)
