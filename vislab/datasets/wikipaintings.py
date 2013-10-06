"""
Code to scrape the WikiPaintings website to construct a dataset.
"""

import os
import requests
import bs4
import pandas as pd
import numpy as np

import vislab

DB_NAME = 'wikipaintings'


def get_image_url_for_id(image_id):
    # TODO: switch to using a DB instead of loading DataFrame.
    filename = vislab.config['paths']['shared_data'] + '/wikipaintings_urls.h5'

    if not os.path.exists(filename):
        df = get_detailed_dataset()
        dfs = df['image_url']
        dfs.to_hdf(filename, 'df', mode='w')
    else:
        dfs = pd.read_hdf(filename, 'df')

    assert(image_id in dfs.index)
    return dfs.ix[image_id]


def get_style_dataset(min_pos=1000):
    """
    Return DataFrame that is suitable for learning style and genre
    properties of artworks.

    Filter on:
    - have more than min_pos examples,
    - have both a style and a genre.

    Expand the style labels to own columns, with 'style_' prefix.
    Expand the genre labels to own columns, with 'genre_' prefix.
    """
    df = get_detailed_dataset()
    df = df.dropna(subset=['style'])
    df = df.dropna(subset=['genre'])

    def filter_and_expand(df, prop, min_pos=None):
        freqs = df[prop].value_counts()

        # Filter out vals with less than min_pos examples.
        if min_pos is not None:
            freqs = freqs[freqs >= min_pos]
        prop_vals = freqs.index.tolist()
        df = df[df[prop].apply(lambda x: x in prop_vals)]

        # Expand into own columns, prefixed by property name.
        for val in prop_vals:
            ascii_name = val.replace(' ', '_').encode('ascii', 'ignore')
            df['{}_{}'.format(prop, ascii_name)] = (df[prop] == val)
        return df

    df = filter_and_expand(df, 'style', min_pos)
    df = filter_and_expand(df, 'genre', min_pos)
    return df


def get_basic_dataset(force=False):
    """
    Return DataFrame of image_id -> page_url, artist_slug, artwork_slug.
    """
    filename = vislab.config['paths']['shared_data'] + \
        '/wikipaintings_basic_info.h5'
    df = vislab.util.load_or_generate_df(filename, fetch_basic_dataset, force)
    return df


def fetch_basic_dataset():
    """
    Fetch basic info and page urls of all artworks by crawling search
    results.
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


def get_detailed_dataset(force=False):
    """
    Return DataFrame of image_id -> detailed artwork info, including
    image URLs.
    """
    filename = vislab.config['paths']['shared_data'] + \
        '/wikipaintings_detailed_info.h5'
    df = vislab.util.load_or_generate_df(
        filename, fetch_detailed_dataset, force)
    return df


def fetch_detailed_dataset(
        force=False, num_workers=1, mem=2000,
        cpus_per_task=1, async=True):
    """
    Fetch detailed info by crawling the detailed artwork pages, using
    the links from the basic dataset.
    """
    print("Fetching detailed Wikipaintings dataset by scraping artwork pages.")

    db = vislab.util.get_mongodb_client()[DB_NAME]
    collection = db['image_info']
    print("Old collection size: {}".format(collection.count()))

    basic_df = get_basic_dataset()
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
        num_workers=num_workers, mem=mem, cpus_per_task=cpus_per_task,
        async=async)
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


if __name__ == '__main__':
    """
    Run the scraping with a number of workers taking jobs from a queue.
    """
    fetch_detailed_dataset(force=True, num_workers=20, async=True)
