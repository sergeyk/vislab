"""
Copyright Sergey Karayev - 2013.

Scrape wikipaintings.org to construct a dataset.
"""
import os
import requests
import bs4
import pandas as pd
import numpy as np
import vislab
import vislab.dataset

DB_NAME = 'wikipaintings'

_DIRNAME = vislab.config['paths']['shared_data']
BASIC_DF_FILENAME = _DIRNAME + '/wikipaintings_basic_info.h5'
DETAILED_DF_FILENAME = _DIRNAME + '/wikipaintings_detailed_info.h5'
URL_DF_FILENAME = _DIRNAME + '/wikipaintings_urls.h5'


def get_image_url_for_id(image_id):
    filename = URL_DF_FILENAME
    if not os.path.exists(filename):
        df = get_df()
        dfs = df['image_url']
        dfs.to_hdf(filename, 'df', mode='w')
    else:
        dfs = pd.read_hdf(filename, 'df')
    return dfs.ix[image_id]


def get_basic_df(force=False, args=None):
    """
    Return DataFrame of image_id -> detailed artwork info, including
    image URLs.
    """
    return vislab.util.load_or_generate_df(
        BASIC_DF_FILENAME, _fetch_basic_dataset, force, args)


def get_df(force=False, args=None):
    """
    Return DataFrame of image_id -> detailed artwork info, including
    image URLs.
    Only keep those artworks with listed style and genre.
    """
    df = vislab.util.load_or_generate_df(
        DETAILED_DF_FILENAME, _fetch_detailed_dataset, force, args)
    df = df.dropna(how='any', subset=['genre', 'style'])
    return df


def get_style_df(min_positive_examples=1000, force=False):
    df = get_df(force)
    return _get_column_label_df(df, 'style', min_positive_examples)


def get_genre_df(min_positive_examples=1000, force=False):
    df = get_df(force)
    return _get_column_label_df(df, 'genre', min_positive_examples)


def get_artist_df(min_positive_examples=200, force=False):
    df = get_df(force)
    df['artist'] = df['artist_slug']
    return _get_column_label_df(df, 'artist', min_positive_examples)


def _get_column_label_df(
        df, column_name, min_positive_examples=500):
    bool_df = vislab.dataset.get_bool_df(
        df, 'style', min_positive_examples)
    bool_df['_split'] = vislab.dataset.get_train_test_split(bool_df)
    bool_df['image_url'] = df['image_url']
    return bool_df


def _fetch_basic_dataset(args=None):
    """
    Fetch basic info and page urls of all artworks by crawling search
    results.  Results are returned as a DataFrame.

    Not parallelized.
    """
    print("Fetching basic Wikipaintings dataset by scraping search results.")

    search_url = 'http://www.wikipaintings.org/en/search/Any/{}'

    # Manual inspection of the available results on 20 Sep 2013
    # showed 1894 pages, with a blank last page, and reportedly
    # 113615 artworks and 1540 artists.
    all_links = []
    for page in range(1, 1894):
        url = search_url.format(page)
        try:
            links = _get_links_from_search_results(url)
        except:
            pass
        all_links += links
        page += 1

    # Turn URLs into image ids and get other basic info.
    df = pd.DataFrame([
        {
            'page_url': 'http://www.wikipaintings.org' + slug,
            'image_id': slug.replace('/en/', '').replace('/', '_'),
            'artist_slug': slug.split('/')[-2],
            'artwork_slug':slug.split('/')[-1]
        } for slug in all_links
    ])
    df.index = pd.Index(df['image_id'], name='image_id')
    return df


def _fetch_detailed_dataset(args=None):
    """
    Fetch detailed info by crawling the detailed artwork pages, using
    the links from the basic dataset.

    Parallelized with vislab.utils.distributed.map_through_rq.
    """
    basic_df = get_basic_df(args)

    print("Fetching detailed Wikipaintings dataset by scraping artwork pages.")

    if args is None:
        args = {
            'force_dataset': False,
            'num_workers': 1, 'mem': 2000,
            'cpus_per_task': 1, 'async': True
        }

    db = vislab.util.get_mongodb_client()[DB_NAME]
    collection = db['image_info']
    print("Old collection size: {}".format(collection.count()))

    force = args.force_dataset
    if not force:
        # Exclude ids that were already computed.
        image_ids = basic_df.index.tolist()
        image_ids = vislab.util.exclude_ids_in_collection(
            image_ids, collection)
        basic_df = basic_df.ix[image_ids]

    # Chunk up the rows.
    rows = [row.to_dict() for ind, row in basic_df.iterrows()]
    chunk_size = 10
    num_chunks = len(rows) / chunk_size
    chunks = np.array_split(rows, num_chunks)
    args_list = [(chunk.tolist(), force) for chunk in chunks]

    # Work the jobs.
    vislab.utils.distributed.map_through_rq(
        vislab.datasets.wikipaintings._fetch_artwork_infos,
        args_list, 'wikipaintings_info',
        num_workers=args['num_workers'], mem=args['mem'],
        cpus_per_task=args['cpus_per_task'], async=args['async'])
    print("Final collection size: {}".format(collection.count()))

    # Assemble into DataFrame to return.
    # Drop artworks without an image.
    orig_df = pd.DataFrame([doc for doc in collection.find()])
    df = orig_df.dropna(subset=['image']).copy()

    # Rename some columns and add an index.
    df['image_url'] = df['image']
    df['date'] = df['dateCreated']
    df.index = pd.Index(df['image_id'], name='image_id')

    # Only take useful columns.
    columns_to_take = [
        'image_id', 'artist_slug', 'artwork_slug', 'date',
        'genre', 'style', 'technique', 'keywords', 'name',
        'page_url', 'image_url'
    ]
    df = df[columns_to_take]

    # Drop artworks with messed up image urls
    good_inds = []
    for ind, row in df.iterrows():
        try:
            str(row['image_url'])
            good_inds.append(ind)
        except:
            pass
    df = df.ix[good_inds]
    df['image_url'] = df['image_url'].apply(lambda x: str(x))

    return df


def _get_links_from_search_results(url):
    try:
        r = requests.get(url)
        soup = bs4.BeautifulSoup(r.text)
    except Exception as e:
        raise e
    links = []
    for item in soup.findAll('ins', class_='search-item'):
        links.append(item.a.attrs['href'])
    return links


def _fetch_artwork_infos(image_ids_and_page_urls, force=False):
    """
    Fetch artwork info, including image url, from the artwork page for
    each of the given image_ids, storing the obtained info to DB.
    """
    collection = vislab.util.get_mongodb_client()[DB_NAME]['image_info']
    collection.ensure_index('image_id')

    for row in image_ids_and_page_urls:
        if not force:
            # Skip if image exists in the database.
            cursor = collection.find({'image_id': row['image_id']})
            if cursor.limit(1).count() > 0:
                continue

        # Get detailed info for the image.
        info = _fetch_artwork_info(row['image_id'], row['page_url'])
        info.update(row)

        collection.update(
            {'image_id': info['image_id']}, info, upsert=True)
        print('inserted {}'.format(info['image_id']))


def _fetch_artwork_info(image_id, page_url):
    """
    Scrape the artwork info page for relevant properties to return dict.
    """
    r = requests.get(page_url)
    soup = bs4.BeautifulSoup(r.text)
    info = {}
    for tag in soup.findAll(lambda tag: 'itemprop' in tag.attrs):
        itemprop = tag.attrs['itemprop']
        info[itemprop] = tag.text
        if itemprop == 'keywords':
            info[itemprop] = info[itemprop].strip().split(',')
        if tag.name == 'img':
            info[itemprop] = tag.attrs['src'].split('!')[0]
    return info
