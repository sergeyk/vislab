"""
Copyright Sergey Karayev / Adobe - 2013.
Written during internship at Adobe CTL, San Francisco.

Code to load the AVA dataset.
Download it from http://www.lucamarchesotti.com/AVA/

The image urls are not part of the above release, and have to be
scraped from dpchallenge.net.
"""
import pandas as pd
import numpy as np
import requests
import bs4
import vislab
import vislab.utils
import vislab.utils.distributed


## Paths
AVA_PATH = vislab.config['paths']['AVA']
AVA_DF_FILENAME = vislab.config['paths']['shared_data'] + '/ava.h5'
STYLE_DF_FILENAME = vislab.config['paths']['shared_data'] + '/ava_style.h5'
URL_DF_FILENAME = vislab.config['paths']['shared_data'] + '/ava_urls.h5'

AVA_URL_FOR_ID = 'http://www.dpchallenge.com/image.php?IMAGE_ID={}'


## Caching loaders of the DataFrames.
def get_urls_df(force=False, args=None):
    """
    Return DataFrame of image_url and page_url.

    NOTE: takes about .4 sec to read from HDF cache.
    """
    df = vislab.util.load_or_generate_df(
        URL_DF_FILENAME, _scrape_image_urls, force, args)
    df['page_url'] = [AVA_URL_FOR_ID.format(ind) for ind in df.index]
    return df


def get_ava_df(force=False, args=None):
    """
    Return DataFrame of the basic AVA dataset.
    """
    return vislab.util.load_or_generate_df(
        AVA_DF_FILENAME, _load_ava_df, force, args)


def get_ratings_df(force=False, args=None):
    """
    Return DataFrame of raw and binarized ratings mean and std.
    """
    df = get_ava_df(force, args)

    # Challenge-binarized mean and std: how does the image compare to
    # the mean of its challenge?
    grouped_df = df.groupby('challenge_name').mean()
    grouped_df.columns = ['challenge_rating_mean', 'challenge_rating_std']
    df = df.join(grouped_df, on='challenge_name')
    df['rating_mean_cn_bin'] = df['rating_mean'] > df['challenge_rating_mean']
    df['rating_std_cn_bin'] = df['rating_std'] > df['challenge_rating_std']

    # Global-binarized mean and std: how does the image compare to
    # the global mean?
    df['rating_mean_bin'] = df['rating_mean'] > df['rating_mean'].mean()
    df['rating_std_bin'] = df['rating_std'] > df['rating_std'].mean()

    return df[[
        'rating_mean', 'rating_std',
        'rating_mean_bin', 'rating_std_bin',
        'rating_mean_cn_bin', 'rating_std_cn_bin'
    ]]


def get_style_df(force=False, args=None):
    """
    Return DataFrame of the AVA style labels: column per label.
    """
    style_df = vislab.util.load_or_generate_df(
        STYLE_DF_FILENAME, _load_style_df, force, args)
    ratings_df = vislab.datasets.ava.get_ratings_df()
    return style_df.join(ratings_df)


## From-scratch loaders.
def _load_ava_df(args=None):
    """
    Load the whole AVA dataset from files as a DataFrame, with columns
        image_id: string
        ratings: list of ints,
        ratings_mean: float,
        ratings_std: float,
        semantic_tag_X_name:string (for X in [1, 2]),
        challenge_name: string.

    Returns
    -------
    df: pandas.DataFrame
    """
    print("load_ava_dataset: loading from files")

    def load_ids_and_names(filename, column_name):
        with open(filename, 'r') as f:
            # example of an (id, name) line: "37 Diptych / Triptych"
            data = [
                (int(line.split()[0]), ' '.join(line.split()[1:]))
                for line in f
            ]
        ids, names = zip(*data)
        return pd.Series(list(names), list(ids), dtype=str)

    # Load the tag and challenge id-name mapping.
    tags = load_ids_and_names(
        AVA_PATH + '/tags.txt', 'semantic_tag_name')
    challenges = load_ids_and_names(
        AVA_PATH + '/challenges.txt', 'challenge_name')

    # Load the main data.
    f = AVA_PATH + '/AVA.txt'
    X = pd.read_csv(f, sep=' ', header=None).values.T

    # Compute the mean and standard deviation of the ratings distributions.
    ratings = X[2:12].T
    values = np.arange(1, 11.)
    ratings_sums = ratings.sum(1)
    rating_mean = (ratings * values).sum(1) / ratings_sums
    rating_std = np.array([np.repeat(values, row).std() for row in ratings])

    # Make DataFrame, using the challenge and semantic tag names
    # instead of ids.
    index = pd.Index(X[1].astype(str), name='image_id')
    df = pd.DataFrame({
        'ratings': [row for row in ratings],
        'rating_mean': np.round(rating_mean, 4),
        'rating_std': np.round(rating_std, 4),
        'semantic_tag_1_name': tags.ix[X[12]].values,
        'semantic_tag_2_name': tags.ix[X[13]].values,
        'challenge_name': challenges.ix[X[14]].values
    }, index)

    return df


def _load_style_df(args=None):
    """
    Load the provided style label information as a DataFrame.
    The provided style labels are split between train and test sets,
    where the latter is multi-label.

    We load the styles into one DataFrame.
    """
    # Load the style category labels
    d = AVA_PATH + '/style_image_lists'
    names_df = pd.read_csv(
        d + '/styles.txt', sep=' ', index_col=0, names=['style'])
    styles = names_df.values.flatten()

    # Load the multi-label encoded test data.
    test_df = pd.DataFrame(
        index=np.loadtxt(d + '/test.jpgl', dtype=str),
        data=np.loadtxt(d + '/test.multilab', dtype=bool),
        columns=styles)
    test_df['_split'] = 'test'

    # Load the single-label encoded training data
    train_df = pd.DataFrame(
        index=np.loadtxt(d + '/train.jpgl', dtype=str),
        data={'style_ind': np.loadtxt(d + '/train.lab', dtype=int)})
    train_df = train_df.join(names_df, on='style_ind')
    train_df['_split'] = 'train'

    # Expand the single-label to multi-label encoding.
    for style in styles:
        train_df[style] = False
        index = train_df.index[train_df['style'] == style]
        train_df[style].ix[index] = True

    # Join the two multi-label encoded dataframes, and get rid of extra columns
    df = train_df.append(test_df)[styles.tolist() + ['_split']]

    # Append 'style_' to all style column names.
    df.columns = [
        x if x.startswith('_') else 'style_' + x
        for x in df.columns
    ]

    df.index = df.index.astype(str)
    df.index.name = 'image_id'

    return df


def _scrape_image_urls(args):
    """
    Construct a mapping from AVA image id to actual URL containing the image.
    This is run in parallel, using rq.

    Parameters
    ----------
    args: from argparse
    """
    if args is None:
        args = {
            'num_workers': 8,
            'chunk_size': 20
        }

    df = get_ava_df()
    image_ids = df.index.tolist()

    # TODO: only submit jobs that have not already been done
    vislab.utils.distributed.map_through_rq(
        function=get_image_url_for_id,
        args_list=[(x,) for x in image_ids],
        name="map_ava_ids_to_urls",
        num_workers=args['num_workers'],
        chunk_size=args['chunk_size'])

    collection = _get_url_mongodb_collection()
    image_ids, urls = zip(*[
        (x['image_id'], x['url'])
        for x in collection.find()
    ])
    df = pd.DataFrame({'image_url': list(urls)}, list(image_ids))
    df = df.drop_duplicates()
    df.index.name = 'image_id'

    df['page_url'] = [
        AVA_URL_FOR_ID.format(x) for x in df.index
    ]

    return df


def get_image_url_for_id(image_id):
    """
    Scrape the dpchallenge page for a given id for the actual image URL.
    Store into a MongoDB collection.

    Parameters
    ----------
    image_id: string

    Returns
    -------
    img_url: string or None
        If the website reports that this id is invalid, return None.

    Raises
    ------
    Exception
        If the request or the BS parse did not complete succesfully.
        If no image found, yet website did not report invalid id.
        If multiple image candidates found.
    """

    image_id = str(image_id)

    collection = _get_url_mongodb_collection()
    collection.ensure_index('image_id')
    cursor = collection.find({'image_id': image_id}).limit(1)
    if cursor.count() > 0:
        return cursor[0]['url']

    url = AVA_URL_FOR_ID.format(image_id)
    try:
        r = requests.get(url)
        soup = bs4.BeautifulSoup(r.text)
    except Exception as e:
        raise e

    imgs = soup.findAll(
        lambda tag:
        'alt' in tag.attrs and
        'src' in tag.attrs and
        tag.attrs['src'].startswith('http://images.dpchallenge.com/')
        and 'style' in tag.attrs and
        tag.attrs['src'].find('thumb') < 0
    )
    if len(imgs) < 1:
        if soup.find(text='Invalid IMAGE_ID provided.') is not None:
            return None
        raise Exception('No image found at url {}'.format(url))
    elif len(imgs) > 1:
        raise Exception('More than one image found at url {}'.format(url))

    img_url = imgs[0]['src']
    collection.insert({'image_id': image_id, 'url': img_url})

    return img_url


def _get_url_mongodb_collection():
    """
    Return the MongoDB collection where image URLs are stored.
    """
    return vislab.util.get_mongodb_client()['datasets']['ava_image_urls']


def cmdline_get_urls_df():
    """
    Get the image URLs by scraping dpchallenge.net pages.
    Should do in parallel.
    """
    args = vislab.utils.cmdline.get_args(
        'ava', 'cmdline_get_urls_df', ['datasets', 'processing'])
    force = args['force_dataset']
    get_urls_df(force, args)


if __name__ == '__main__':
    possible_functions = {
        'get_urls_df': cmdline_get_urls_df,
    }
    vislab.util.cmdline.run_function_in_file(__file__, possible_functions)
