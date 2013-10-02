"""
Construct dicts containing identifying information and DataFrames
for train, val, test splits of data, for use with classifiers.
"""
import argparse
import numpy as np
import pandas as pd
import vislab
import aphrodite


def get_importance(df):
    """
    Get importance weights of data points based on the prevalence of the
    label. The most frequent label gets weight less than 1.

    Parameters
    ----------
    df: pandas.DataFrame
        Must have column 'label.'

    Returns
    -------
    importances: pandas.Series
    """
    counts = [
        (df['label'] == label).sum()
        for label in df['label'].unique()
    ]
    mfl = df['label'].unique()[np.argmax(counts)]

    importances = np.ones(df.shape[0])
    ind = (df['label'] == mfl)
    importances[ind] = 1. * (~ind).sum() / ind.sum()

    importances = pd.Series(importances, df.index)
    return importances


def get_binary_dataset(
        df, dataset_name, style_name, random_seed=42, min_pos_frac=.1):
    """
    # TODO: currently assuming that negative data is more prevalent than pos.

    Parameters
    ----------
    dataset_name: string
        In ['flickr', 'wikipaintings']
    style_name: string
        Name of column to take.
    min_pos_frac: float
        Subsample negative data such that pos/neg ratio is at least this.
    """
    df = pd.DataFrame(index=df.index, data={'label': df[style_name]})

    # The total number is the number of + examples.
    num_total = df['label'].sum()
    num_test = int(0.2 * num_total)
    num_val = num_test

    # Equal number positive and negative for test and val sets.
    np.random.seed(random_seed)
    ind = np.random.permutation(num_total)
    test_ids = np.concatenate((
        df[df['label']].index[ind[:num_test]],
        df[~df['label']].index[ind[:num_test]]
    ))

    val_ids = np.concatenate((
        df[df['label']].index[ind[num_test:num_test + num_val]],
        df[~df['label']].index[ind[num_test:num_test + num_val]]
    ))

    train_ids = df.index.diff(test_ids.tolist() + val_ids.tolist())

    # Subsample negative data to respect min_pos_frac.
    train_df = df.ix[train_ids]
    max_num = 1. / min_pos_frac * train_df['label'].sum()
    train_ids = np.concatenate((
        train_df[train_df['label']].index,
        train_df[~train_df['label']].index[:max_num]))

    # Add the remaining ids to the test set, to ensure that all images
    # in the dataset are classified.
    remaining_ids = df.index.diff(
        test_ids.tolist() + val_ids.tolist() + train_ids.tolist())
    test_ids = np.concatenate((test_ids, remaining_ids))

    # Convert to +1/-1 labels.
    labels = np.ones(df.shape[0])
    labels[~df['label']] = -1
    df['label'] = labels
    df = df[['label']]

    # Get the datasets.
    def get_split_df(ids):
        split_df = df.ix[ids]
        split_df['importance'] = get_importance(split_df)
        return split_df

    dataset = {
        'train_df': get_split_df(train_ids),
        'val_df': get_split_df(val_ids),
        'test_df': get_split_df(test_ids)
    }

    # Add all relevant info to the data dict to return.
    dataset.update({
        'dataset_name': dataset_name,
        'name': '{}_style_{}_train_{}'.format(
            dataset_name, style_name, train_df.shape[0]),
        'task': 'clf',
        'num_labels': 2,
        'salient_parts': {
            'data': '{}_{}'.format(dataset_name, style_name),
            'num_train': dataset['train_df'].shape[0],
            'num_val': dataset['val_df'].shape[0],
            'num_test': dataset['test_df'].shape[0]
        }
    })

    return dataset


def add_cmdline_args(parser):
    """
    Add relevant command line arguments to the given ArgumentParser.
    Modifies the passed-in object.

    Parameters
    ----------
    parser: argparse.ArgumentParser
    """
    parser.add_argument(
        '--force_dataset',
        help="force load of the dataset from source instead of cache",
        action="store_true", default=False)
    parser.add_argument(
        '--dataset',
        help="select which dataset to use",
        required=True,
        choices=['ava', 'ava_style', 'flickr', 'wikipaintings'])
    parser.add_argument(
        '--num_images',
        help="number of images to use from the dataset (-1 for all)",
        type=int, default=-1)


def load_dataset_with_args(args):
    """
    Use the parsed command line arguments to load the correct dataset.
    """
    if args.dataset == 'ava':
        df = aphrodite.dataset.load_ava_df(
            args.num_images, args.random_seed, args.ava_force)

    elif args.dataset == 'ava_style':
        style_df = aphrodite.dataset.load_style_df()
        df = df.ix[style_df.index]

    elif args.dataset == 'flickr':
        df = aphrodite.flickr.load_flickr_df(
            args.num_images, args.random_seed, args.dataset_force)

    elif args.dataset == 'wikipaintings':
        df = vislab.datasets.wikipaintings.get_style_dataset(min_pos=1000)
        if args.num_images > 0:
            df = df.iloc[:args.num_images]

    else:
        raise Exception('Unknown dataset.')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a dataset.")
    vislab.util.add_cmdline_args(parser)
    vislab.utils.distributed.add_cmdline_args(parser)
    add_cmdline_args(parser)
    args = parser.parse_args()
