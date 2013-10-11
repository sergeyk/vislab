"""
Code to load the AVA dataset.

Static files for the dataset should be placed into
config['paths']['static_data']
"""
import os
import pandas as pd
import numpy as np
import aphrodite
import aphrodite.image
import vislab


AVA_PATH = vislab.config['paths']['static_data'] + '/AVA_dataset'
# TODO: get rid of the below, load urls into dataframe once
URL_MAP_FILENAME = aphrodite.REPO_DIRNAME + '/data_static/ava_id_to_image_url_map.csv.gz'


def get_image_urls():
    df = pd.read_csv(
        URL_MAP_FILENAME, compression='gzip', index_col=0)
    df.index = df.index.astype(str)
    df['page_url'] = [aphrodite.image.AVA_URL_FOR_ID.format(x) for x in df.index]
    return df


def get_ava_dataset(force=False):
    filename = vislab.config['paths']['shared_data'] + '/ava.h5'
    df = vislab.util.load_or_generate_df(filename, load_ava_dataset, force)
    return df


def load_ava_dataset():
    """
    Load the whole AVA dataset from files as a DataFrame, with columns
        image_id: string
        ratings: list of ints,
        semantic_tag_X_id:int
        semantic_tag_X_name:string (for X in [1, 2]),
        challenge_id: int
        challenge_name: string.

    Returns
    -------
    df: pandas.DataFrame
    """
    def load_ids_and_names(filename, column_name):
        with open(filename, 'r') as f:
            lines = f.readlines()
        # example of an (id, name) line: "37 Diptych / Triptych"
        data = [(int(l.split()[0]), ' '.join(l.split()[1:])) for l in lines]
        ids, names = zip(*data)
        df = pd.DataFrame(
            data=list(names), index=list(ids),
            columns=[column_name], dtype=str)
        return df

    # Load the tag and challenge id-name mapping.
    tags_df = load_ids_and_names(
        AVA_PATH + '/tags.txt', 'semantic_tag_name')
    challenges_df = load_ids_and_names(
        AVA_PATH + '/challenges.txt', 'challenge_name')

    # Load the main data.
    f = AVA_PATH + '/AVA.txt'
    X = pd.read_csv(f, sep=' ', header=None).values.T
    ratings = X[2:12].T
    rating_mean = (
        np.arange(1, 11.) * ratings / ratings.sum(1)[:, np.newaxis]).sum(1)
    rating_std = np.array(list(np.repeat(np.arange(1, 11.), row).std() for row in ratings))
    df = pd.DataFrame({
        'ratings': [row for row in ratings],
        'rating_mean': rating_mean, 'rating_std': rating_std,
        'semantic_tag_1_id': X[12], 'semantic_tag_2_id': X[13],
        'challenge_id': X[14]
    }, index=X[1].astype(str))

    df.rating_mean = np.round(df.rating_mean, 4)
    df.rating_std = np.round(df.rating_std, 4)

    # Store the names of the tags and challenges along with the ids.
    df['semantic_tag_1_name'] = df.join(
        tags_df, on='semantic_tag_1_id', how='left')['semantic_tag_name']
    df['semantic_tag_2_name'] = df.join(
        tags_df, on='semantic_tag_2_id', how='left')['semantic_tag_name']
    df = df.join(challenges_df, on='challenge_id', how='left')

    df.index.name = 'image_id'

    return df


def load_style_df(force=False):
    """
    Load the provided style label information as a DataFrame.
    The provided style labels are split between train and test sets,
    where the latter is multi-label.
    """
    filename = vislab.config['paths']['shared_data'] + '/ava_style.h5'
    if not force and os.path.exists(filename):
        return pd.read_hdf(filename, 'df')

    # Load the style category labels
    d = AVA_PATH + '/style_image_lists'
    names_df = pd.read_csv(
        d + '/styles.txt', sep=' ', index_col=0, names=['Style'])
    styles = names_df.values.flatten()

    # Load the multi-label encoded test data.
    test_df = pd.DataFrame(
        index=np.loadtxt(d + '/test.jpgl', dtype=str),
        data=np.loadtxt(d + '/test.multilab', dtype=bool),
        columns=styles)

    # Load the single-label encoded training data
    train_df = pd.DataFrame(
        index=np.loadtxt(d + '/train.jpgl', dtype=str),
        data={'style_ind': np.loadtxt(d + '/train.lab', dtype=int)})
    train_df = train_df.join(names_df, on='style_ind')

    # Expand the single-label to multi-label encoding.
    for style in styles:
        train_df[style] = False
        index = train_df.index[train_df['Style'] == style]
        train_df[style].ix[index] = True

    # Joni the two multi-label encoded dataframes, and get rid of extra columns
    df = train_df.append(test_df)[styles]

    df.index.name = 'image_id'

    df.to_hdf(filename, 'df')
    return df


def get_style_dataset(ava_df, random_seed=42):
    """
    Styles are [
        'Complementary_Colors', 'Duotones', 'HDR', 'Image_Grain',
        'Light_On_White', 'Long_Exposure', 'Macro', 'Motion_Blur',
        'Negative_Image', 'Rule_of_Thirds', 'Shallow_DOF', 'Silhouettes',
        'Soft_Focus', 'Vanishing_Point']

    The split is as given in the AVA dataset, although it's unclear to me
    if our test procedure is exactly the same.
    """
    d = AVA_PATH + '/style_image_lists'
    names_df = pd.read_csv(
        d + '/styles.txt', sep=' ', index_col=0, names=['Style'])

    trainval_df = pd.DataFrame(
        index=np.loadtxt(d + '/train.jpgl', dtype=str),
        data={'label': np.loadtxt(d + '/train.lab', dtype=int)})
    trainval_df['importance'] = 1

    num_labels = len(trainval_df['label'].unique())
    task = 'clf'

    test_df = pd.DataFrame(
        index=np.loadtxt(d + '/test.jpgl', dtype=str),
        data=np.loadtxt(d + '/test.multilab', dtype=bool))
    ind, style_ind = np.where(test_df)
    test_df_unrolled = pd.DataFrame(
        index=test_df.index[ind], data={'label': style_ind + 1})
    test_df_unrolled['importance'] = 1

    # Shuffle training data and make a validation set.
    num_train = int(0.7 * trainval_df.shape[0])
    shuffle_ind = np.random.permutation(trainval_df.shape[0])
    train_df = trainval_df.iloc[shuffle_ind[:num_train]]
    val_df = trainval_df.iloc[shuffle_ind[num_train:]]

    name = 'clf_style_train_{}_test_{}'.format(
        train_df.shape[0], test_df_unrolled.shape[0])

    dataset = {
        'dataset_name': 'ava',
        'name': name,
        'task': task,
        'salient_parts': {
            'data': 'style',
            'num_train': train_df.shape[0],
            'num_val': val_df.shape[0],
            'num_test': test_df_unrolled.shape[0]
        },
        'names_df': names_df,
        'num_labels': num_labels,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df_unrolled
    }
    return dataset


def construct_label_df(df, ids, balance=False):
    """
    Construct DataFrames with scores and labels.
    If balance==True, equalize the number of positive and negative examples.
    """
    labels = df['label'].ix[ids]

    # most frequent label
    mfl = 1
    if (labels == -1).sum() > (labels == 1).sum():
        mfl = -1

    importances = np.ones(len(labels))
    if balance:
        ind = np.random.permutation((labels == mfl).sum())
        ind = ind[:(labels != mfl).sum()]
    else:
        importances[labels == mfl] = (
            1. * (labels != mfl).sum() / (labels == mfl).sum())

    df = pd.DataFrame(
        data={'importance': importances, 'label': labels}, index=ids)

    if balance:
        df2 = df[labels != mfl]
        df2 = df2.append(df[labels == mfl].iloc[ind])
        df = df2

    # Sort by id to have compatibility with feature database.
    df = df.ix[sorted(df.index)]

    return df


def get_rating_dataset(
        ava_df, task='clf_rating_mean', frac_test=.2, num_train=-1, frac_val=-1,
        delta=0, random_seed=42):
    """
    Return ids, scores, and labels dictionary for prediction of
    'attractiveness' of a photograph.
    Depending on the task, the label can be used for classification or
    regression of either the mean or the standard deviation of the ratings.

    Parameters
    ----------
    ava_df: pandas.DataFrame

    task: string in [
        'clf_rating_mean', 'clf_rating_std',
        'regr_rating_mean', 'regr_rating_std'
    ]

    frac_test: int [.3]
        Fraction of images to use for testing.

    num_train: int [-1]
        If < 0, use all remaining images after num_test is subtracted.

    frac_val: int [-1]
        Fraction of images to use in validation: these come from the training set.
        If -1, use the same number as test images.

    delta: float [0]
        If > 0,  discard from the training set all images with average
        score in [rating_mean - delta/2, rating_mean + delta/2].

    random_seed: float [42]

    Returns
    -------
    dataset: dict
        Contains 'train_ids', 'train_scores', 'train_labels',
                 'test_ids', 'test_scores', 'test_labels'
    """
    assert(delta >= 0)
    num_images = ava_df.shape[0]

    num_test = int(frac_test * num_images)
    assert(num_test > 0 and num_test < num_images)

    if frac_val < 0:
        num_val = num_test
    else:
        num_val = int(frac_val * num_images)

    # Set the label
    ava_df = ava_df.copy()
    rating_mean_mean = ava_df['rating_mean'].mean()
    rating_std_mean = ava_df['rating_std'].mean()
    if task == 'clf_rating_mean':
        num_labels = 2
        task_ = 'clf'
        ava_df['label'] = -1
        ava_df['label'][ava_df['rating_mean'] > rating_mean_mean] = 1
    elif task == 'clf_rating_std':
        num_labels = 2
        task_ = 'clf'
        ava_df['label'] = -1
        ava_df['label'][ava_df['rating_std'] > rating_std_mean] = 1
    elif task == 'regr_rating_mean':
        num_labels = -1
        task_ = 'regr'
        ava_df['label'] = ava_df['rating_mean']
    elif task == 'regr_rating_std':
        num_labels = -1
        task_ = 'regr'
        ava_df['label'] = ava_df['rating_std']
    else:
        raise Exception("Unimplemented task")

    # Select test images, and leave the rest as train candidates.
    if num_train < 0 or num_train > (num_images - num_test):
        num_train = num_images - num_test

    np.random.seed(random_seed)
    ind = np.random.permutation(num_images)
    test_ids = ava_df.index[ind[:num_test]]
    val_ids = ava_df.index[ind[num_test:num_test + num_val]]
    train_candidates_df = ava_df.iloc[ind[num_test + num_val:]]

    # Filter on delta.
    if delta > 0:
        train_candidates_df = train_candidates_df[
            (train_candidates_df['rating_mean'] <= rating_mean_mean - delta / 2.) |
            (train_candidates_df['rating_mean'] >= rating_mean_mean + delta / 2.)]
    if train_candidates_df.shape[0] <= 1:
        raise Exception('Not enough training examples. Try lowering delta.')
    if train_candidates_df.shape[0] > num_train:
        ind = np.random.permutation(train_candidates_df.shape[0])
        train_ids = train_candidates_df.index[ind[:num_train]]
    else:
        train_ids = train_candidates_df.index

    train_df = construct_label_df(ava_df, train_ids, balance=False)
    val_df = construct_label_df(ava_df, val_ids, balance=True)
    test_df = construct_label_df(ava_df, test_ids, balance=True)

    name = '{}_train_{}_test_{}_delta_{:.2f}'.format(
        task, num_train, num_test, delta)
    data = '_'.join(task.split('_')[1:])  # like 'rating_mean'

    dataset = {
        'dataset_name': 'ava',
        'name': name,
        'task': task_,
        'num_labels': num_labels,
        'salient_parts': {
            'data': data,
            'num_test': num_test,
            'actual_num_test': test_df.shape[0],
            'num_val': num_val,
            'actual_num_val': val_df.shape[0],
            'num_train': num_train,
            'actual_num_train': train_df.shape[0],
            'delta': delta
        },
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }
    return dataset
