"""
Possible experiments:

- Flickr style, binary
- Flickr style, OAA
- PASCAL, binary
- PASCAL, OAA
- Wikipaintings style, binary
- Wikipaintings style, OAA
- AVA rating_mean, binary
- AVA rating_std, binary
- AVA rating_mean 3-way, OAA
- AVA style, binary
- AVA style, OAA
"""
import copy
import logging
import pandas as pd
import numpy as np
import vislab.utils.cmdline
import vislab.dataset
import vislab.vw


def _process_df_for_regression(df, test_frac):
    N = df.shape[0]
    num_test = int(round(test_frac * N))
    num_val = num_test

    ind = np.random.permutation(N)
    test_ids = df.index[ind[:num_test]]
    val_ids = df.index[ind[num_test:num_test + num_val]]
    train_ids = df.index[ind[num_test + num_val:]]

    return df, train_ids, test_ids, val_ids


def _process_df_for_binary_clf_with_split(
        df, split_series, test_frac, min_pos_frac):
    """
    Respect the given split information.
    """
    assert(split_series.shape[0] == df.shape[0])

    test_ids = split_series[split_series == 'test'].index

    # Split the remaining ids into train and val.
    # Val should be test_frac of train and be balanced.
    remaining_ids = split_series[split_series != 'test'].index
    rdf = df.ix[remaining_ids]

    N = rdf.shape[0]
    num_total = rdf['label'].sum()
    num_val = int(test_frac * num_total)

    pos_ind = np.random.permutation(num_total)
    neg_ind = np.random.permutation(N - num_total)

    assert(num_val < len(pos_ind))
    assert(num_val < len(neg_ind))

    val_ids = np.concatenate((
        rdf[rdf['label']].index[pos_ind[:num_val]],
        rdf[~rdf['label']].index[neg_ind[:num_val]]
    ))

    train_ids = rdf.index.diff(val_ids.tolist())

    # To respect min_pos_frac, have to subsample negative examples.
    if min_pos_frac > 0:
        train_df = rdf.ix[train_ids]
        max_num = 1. / min_pos_frac * train_df['label'].sum()
        train_ids = np.concatenate((
            train_df[train_df['label']].index,
            train_df[~train_df['label']].index[:max_num]
        ))

    # Convert to +1/-1 labels.
    labels = pd.Series(-np.ones(df.shape[0]), index=df.index)
    labels[df['label']] = 1
    df['label'] = labels.astype(int)

    return df, train_ids, val_ids, test_ids


def _process_df_for_binary_clf(df, test_frac, min_pos_frac):
    # The total number is the number of + examples.
    N = df.shape[0]
    num_total = df['label'].sum()
    num_test = int(round(test_frac * num_total))
    num_val = num_test

    # Take equal number + and - examples for the test and val sets.
    pos_ind = np.random.permutation(num_total)
    neg_ind = np.random.permutation(N - num_total)

    assert(num_test + num_val < len(pos_ind))
    assert(num_test + num_val < len(neg_ind))

    test_ids = np.concatenate((
        df[df['label']].index[pos_ind[:num_test]],
        df[~df['label']].index[neg_ind[:num_test]]
    ))

    val_ids = np.concatenate((
        df[df['label']].index[pos_ind[num_test:num_test + num_val]],
        df[~df['label']].index[neg_ind[num_test:num_test + num_val]]
    ))

    # At first, take all other examples for the training set.
    train_ids = df.index.diff(test_ids.tolist() + val_ids.tolist())

    # But to respect min_pos_frac, have to subsample negative examples.
    train_df = df.ix[train_ids]
    max_num = 1. / min_pos_frac * train_df['label'].sum()
    train_ids = np.concatenate((
        train_df[train_df['label']].index,
        train_df[~train_df['label']].index[:max_num]
    ))

    # Add the remaining ids to the test set, to ensure that all images
    # in the dataset are classified.
    remaining_ids = df.index.diff(
        test_ids.tolist() + val_ids.tolist() + train_ids.tolist())
    test_ids = np.concatenate((test_ids, remaining_ids))

    # Convert to +1/-1 labels.
    labels = pd.Series(-np.ones(df.shape[0]), index=df.index)
    labels[df['label']] = 1
    df['label'] = labels.astype(int)

    return df, train_ids, val_ids, test_ids


def get_split_df(df, ids, num_labels):
    split_df = df.ix[ids]
    split_df['importance'] = _get_importance(split_df, num_labels)
    return split_df


def get_binary_or_regression_dataset(
        source_df, dataset_name, column_name,
        test_frac=.2, min_pos_frac=.1, random_seed=42):
    """
    Return a dataset dict suitable for the prediction code of binary
    or regression data in column_name column of source_df.
    Whether the data is binary or regression is inferred from dtype.

    # TODO: add ability to pass a filter to use for the AVA delta stuff

    Parameters
    ----------
    source_df: pandas.DataFrame
    dataset_name: string
    column_name: string
    test_frac: float
        Use this fraction of the positive examples to test.
        Will use the same amount for validation.
    min_pos_frac: float
        Subsample negative data s.t. pos/neg ratio is at least this.
        Only relevant if the data is binary, obviously.
        Ignored if < 0.
    random_seed: int [42]
    """
    assert(source_df.index.dtype == object)

    np.random.seed(random_seed)

    df = pd.DataFrame(
        {'label': source_df[column_name]}, source_df.index)

    # Establish whether the data is for binary or regression,
    # and split the dataset into train/val/test appropriately.
    unique_labels = df['label'].unique()
    if df['label'].dtype == bool or len(unique_labels) == 2:
        task = 'clf'
        num_labels = 2

        if df['label'].dtype != bool:
            assert(1 in unique_labels and -1 in unique_labels)
            df['label'][df['label'] == 1] = True
            df['label'][df['label'] == -1] = False
            df['label'] = df['label'].astype(bool)

        if '_split' in source_df.columns:
            df, train_ids, val_ids, test_ids = \
                _process_df_for_binary_clf_with_split(
                    df, source_df['_split'], test_frac, min_pos_frac)
        else:
            df, train_ids, val_ids, test_ids = _process_df_for_binary_clf(
                df, test_frac, min_pos_frac)

    elif df['label'].dtype == float:
        task = 'regr'
        num_labels = -1
        df, train_ids, val_ids, test_ids = _process_df_for_regression(
            df, test_frac)

    else:
        raise Exception("Can only deal with binary or float values.")

    # Get the train/val/test datasets.
    dataset = {
        'train_df': get_split_df(df, train_ids, num_labels),
        'val_df': get_split_df(df, val_ids, num_labels),
        'test_df': get_split_df(df, test_ids, num_labels)
    }

    # Add all relevant info to the data dict to return.
    dataset.update({
        'dataset_name': dataset_name,
        'name': '{}_{}_train_{}'.format(
            dataset_name, column_name, dataset['train_df'].shape[0]),
        'task': task,
        'num_labels': num_labels,
        'salient_parts': {
            'data': '{}_{}'.format(dataset_name, column_name),
            'num_train': dataset['train_df'].shape[0],
            'num_val': dataset['val_df'].shape[0],
            'num_test': dataset['test_df'].shape[0]
        }
    })

    return dataset


def get_multiclass_dataset(
        source_df, dataset_name, column_set_name, column_names,
        test_frac=.2, balanced=False, random_seed=42):
    """
    Return a dataset dict for multi-class data.

    TODO: support multi-label: stick them all in test.

    Parameters
    ----------
    source_df: pandas.DataFrame
    dataset_name: string
    column_set_name: string
        Name for the given set of columns. For example, 'artists'
    column_names: sequence of string
    test_frac: float
        Use this fraction of the examples to test.
        Will use the same amount for validation.
    balanced: bool [False]
        If True, val set will have nearly equal class distribution.
    random_seed: int [42]
    """
    assert(source_df.index.dtype == object)
    np.random.seed(random_seed)

    # Drop rows with no positive labels (ava_style has these...)
    ind = source_df[column_names].sum(1) == 0
    logging.info('Dropping {} rows with no positive labels.'.format(ind.sum()))
    source_df = source_df[~ind]

    N = source_df.shape[0]
    num_test = int(round(test_frac * N))
    num_val = num_test
    num_labels = len(column_names)
    task = 'clf'

    # If source_df does not have split info, do the split here.
    if '_split' not in source_df.columns:
        multilabel_ind = np.where(source_df[column_names].sum(1) > 1)[0]
        test_ind = []
        if len(multilabel_ind) > 0:
            # Put all multi-label examples in test, even if
            # test_frac is exceeded.
            test_ind = multilabel_ind.tolist()
        num_remaining = num_test - len(test_ind)
        if num_remaining > 0:
            singlelabel_ind = np.where(source_df[column_names].sum(1) == 1)[0]
            test_ind += np.random.choice(
                singlelabel_ind, num_remaining, replace=False).tolist()
        test_ids = source_df.index[test_ind]

    # Otherwise, just take the given test ids.
    else:
        test_ids = source_df[source_df['_split'] == 'test'].index

    trainval_ids = source_df.index - test_ids

    # Split into single-label trainval and possible multi-label test.
    test_df = source_df[column_names].ix[test_ids]
    # test_df needs dummy 'label' and 'importance' columns for vw_filter
    test_df['label'] = 1
    test_df['importance'] = 1.
    trainval_df = source_df[column_names].ix[trainval_ids]
    assert(np.all(trainval_df.sum(1) == 1))

    # Split trainval into train and val.
    label = trainval_df.values.argmax(1)
    df = pd.DataFrame({'label': label}, trainval_df.index)

    ids = df.index[np.random.permutation(df.shape[0])]

    if balanced:
        # Construct a balanced validation set.
        counts = trainval_df.sum(0).astype(int)
        min_count = counts[counts.argmin()]
        permutation = lambda N, K: np.random.permutation(N)[:K]
        min_size_balanced_set = np.concatenate([
            np.where(df['label'] == l)[0][permutation(count, min_count)]
            for l, count in enumerate(counts)
        ])
        P = min_size_balanced_set.shape[0]
        if P < num_val:
            raise Exception('Not enough balanced data for validation set.')
        min_size_balanced_set = np.random.permutation(min_size_balanced_set)
        val_ids = df.index[min_size_balanced_set[:num_val]]
    else:
        val_ids = ids[:num_val]

    train_ids = ids.diff(val_ids.tolist())
    train_ids = train_ids[np.random.permutation(len(train_ids))]

    # Assert that there is no overlap between the sets.
    assert(len(train_ids.intersection(val_ids)) == 0)
    assert(len(train_ids.intersection(test_ids)) == 0)
    assert(len(val_ids.intersection(test_ids)) == 0)

    # Add 1 to 'label', since VW needs values on [1, K].
    df['label'] += 1

    # Get the train/val/test datasets.
    dataset = {
        'train_df': get_split_df(df, train_ids, num_labels).join(trainval_df),
        'val_df': get_split_df(df, val_ids, num_labels).join(trainval_df),
        'test_df': test_df
    }

    # Add all relevant info to the data dict to return.
    dataset.update({
        'dataset_name': dataset_name,
        'name': '{}_{}_train_{}'.format(
            dataset_name, column_set_name, dataset['train_df'].shape[0]),
        'task': task,
        'num_labels': num_labels,
        'column_names': column_names,
        'salient_parts': {
            'data': '{}_{}'.format(dataset_name, column_set_name),
            'num_train': dataset['train_df'].shape[0],
            'num_val': dataset['val_df'].shape[0],
            'num_test': dataset['test_df'].shape[0]
        }
    })

    return dataset


def _get_importance(df, num_labels):
    """
    Get importance weights of data points. The most frequent label gets
    weight < 1, in proportion to its prevalence.

    Parameters
    ----------
    df: pandas.DataFrame
        Must have column 'label'
    num_labels: int
        If < 0, no importances computed.

    Returns
    -------
    importances: pandas.Series
    """
    importances = pd.Series(np.ones(df.shape[0]), df.index)

    if num_labels > 0:
        counts = df['label'].value_counts()
        mfl_count = float(counts.max())
        for label in counts.index:
            ind = df['label'] == label
            importances[ind] = mfl_count / ind.sum()

    return importances


def get_prediction_dataset_with_args(args, source_df=None):
    """
    args should contain:
        prediction_label: string
            Can contain a prefix followed by a wildcard *: "style_*".
            In that case, all columns starting with prefix are matched.
    """
    df = vislab.dataset.get_df_with_args(args)

    # If we are matching multiple columns, then we need to construct
    # a multi-class dataset.
    if '*' in args.prediction_label:
        column_set_name = args.prediction_label.replace('*', 'ALL')
        prefix = args.prediction_label.split('*')[0]

        # If source_dataset is given, then make a random value DataFrame
        # with target dataset's index and source dataset's columns.
        if source_df is not None:
            column_names = [
                col for col in source_df.columns
                if col.startswith(prefix)
            ]

            # Making dataframe in this way so that only one label is
            # active per row, so that our multiclass-splitting function
            # can deal with it.
            df = vislab.dataset.get_bool_df(
                pd.DataFrame(
                    np.random.choice(column_names, size=df.shape[0]),
                    columns=[''],
                    index=df.index
                ),
                ''
            )

        else:
            column_names = [
                col for col in df.columns if col.startswith(prefix)
            ]

        dataset = get_multiclass_dataset(
            df, args.dataset, column_set_name, column_names,
            args.test_frac, args.balanced, args.random_seed)

    # Otherwise, we are matching either a binary or regression label.
    else:
        if source_df is not None:
            df[args.prediction_label] = np.random.randint(
                2, size=df.shape[0]).astype(bool)

        dataset = get_binary_or_regression_dataset(
            df, args.dataset, args.prediction_label,
            args.test_frac, args.min_pos_frac, args.random_seed)

    return dataset


def predict_from_trained(args=None):
    if args is None:
        args = vislab.utils.cmdline.get_args(
            __file__, 'predict',
            ['dataset', 'prediction', 'processing', 'feature'])

    ## Get the source and target datasets as specified in args.

    # First get the source dataset.
    args_copy = copy.deepcopy(args)
    args_copy.dataset = args_copy.source_dataset
    source_dataset = get_prediction_dataset_with_args(args_copy)

    # Then get the target dataset.
    # If the names are the same, then just use exactly the source.
    if args.dataset == args.source_dataset:
        dataset = copy.deepcopy(source_dataset)

    # If not, then fill with random values.
    else:
        dataset = get_prediction_dataset_with_args(
            args, source_dataset['train_df'])

    vislab.vw.test(
        args.collection_name, dataset, source_dataset, args.features,
        force=args.force_predict, num_workers=args.num_workers,
        bit_precision=args.bit_precision)


def predict(args=None):
    if args is None:
        args = vislab.utils.cmdline.get_args(
            __file__, 'predict',
            ['dataset', 'prediction', 'processing', 'feature'])

    if args.source_dataset is not None:
        return predict_from_trained(args)

    # Get the dataset as specified in args.
    dataset = get_prediction_dataset_with_args(args)

    # If we're doing regression, set the loss function appropriately.
    if dataset['task'] == 'regr':
        loss_functions = ['squared']
    else:
        loss_functions = ['hinge', 'logistic']

    # Set the number of passes. Less passes for quadratic features.
    n_train = dataset['train_df'].shape[0]

    # Rule of thumb: 1M data points, or at least 3 full passes.
    n_iter = max(3, min(n_train, int(np.ceil(1e6 / n_train))))
    num_passes = np.array(sorted(set([n_iter / 3, n_iter]))).astype(int)

    vislab.vw.train_and_test(
        args.collection_name, dataset, args.features,
        force=args.force_predict, num_workers=args.num_workers,
        num_passes=num_passes.tolist(),
        loss=loss_functions,
        l1=[0, 1e-7],
        l2=[0, 1e-7],
        quadratic=args.quadratic,
        bit_precision=args.bit_precision)


if __name__ == '__main__':
    possible_functions = {
        'predict': predict
    }
    print __file__
    vislab.utils.cmdline.run_function_in_file(__file__, possible_functions)
