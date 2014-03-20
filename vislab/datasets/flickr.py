"""
Copyright Sergey Karayev / Adobe - 2013.
Written during internship at Adobe CTL, San Francisco.

Make dataset of images with style labels by querying Flickr Groups using
the Flickr API.

TODO: cache as HDF5, not pickle
"""
import urllib2
import pandas as pd
import numpy as np
import json
import vislab

# Mapping of style names to group ids.
styles = {
    'Geometric Composition': ['46353124@N00'],
    'Macro': ['52241335207@N01'],
    'Depth of Field': ['75418467@N00'],
    'Long Exposure': ['52240257802@N01'],
    'Minimal': ['42097308@N00'],
    'Noir': ['42109523@N00'],
    'Horror': ['29561404@N00'],
    'Melancholy': ['70495179@N00'],
    'Serene': ['1081625@N25'],
    'HDR': ['99275357@N00'],
    'Bright, Energetic': ['799643@N24'],
    'Sunny': ['1242213@N23'],
    'Soft, Pastel': ['1055565@N24'],
    'Ethereal': ['907784@N22'],
    'Romantic': ['54284561@N00'],
    'Hazy': ['38694591@N00'],
    'Vintage': ['1222306@N25'],
}
style_names = styles.keys()
underscored_style_names = [
    'style_' + style.replace(' ', '_') for style in styles.keys()]


def _fetch_df():
    client = vislab.util.get_mongodb_client()
    dfs = []
    for style in style_names:
        df = pd.DataFrame(list(client['flickr'][style].find()))
        df2 = pd.DataFrame(data={
            'image_url': df.apply(lambda row: get_image_url(row), axis=1),
            'page_url': df.apply(lambda row: get_page_url(row), axis=1),
        })
        df2.index = df['image_id'].astype(str)
        style_str = 'style_' + style.replace(' ', '_')
        df2[style_str] = True
        dfs.append(df2)

    main_df = dfs[0]
    for df_ in dfs:
        main_df = main_df.combine_first(df_)
    main_df = main_df.fillna(False)

    # Make sure the values are boolean
    main_df[underscored_style_names] = \
        main_df[underscored_style_names].astype(bool)

    return main_df


def get_df(force=False):
    """
    Load the data from the database into a DataFrame and save it.
    """
    filename = vislab.config['paths']['shared_data'] + '/flickr_df.pickle'
    df = vislab.util.load_or_generate_df(
        filename, _fetch_df, force)

    # Assign split information, and assign the few images that have
    # multiple labels to test.
    df['_split'] = vislab.dataset.get_train_test_split(
        df[underscored_style_names])
    df['_split'][df[underscored_style_names].sum(1) > 1] = 'test'

    return df


def get_image_url(photo, size_flag=''):
    """
    size_flag: string ['']
        See http://www.flickr.com/services/api/misc.urls.html for options.
            '': 500 px on longest side
            '_m': 240px on longest side
    """
    url = "http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg"
    return url.format(size=size_flag, **photo)


def get_page_url(photo):
    url = "http://www.flickr.com/photos/{owner}/{id}"
    return url.format(**photo)


def get_photos_for_style(style, num_images=250):
    """
    Parameters
    ----------
    style: string
    num_images: int [500]
        Fetch at most this many results.
    """
    print('\nget_photos_for_style: {}'.format(style))

    # Establish connection with the MongoDB server.
    client = vislab.util.get_mongodb_client()

    # How many images are already in the database for this style?
    collection = client['flickr'][style]
    collection.ensure_index('image_id')
    collection_size = collection.find({'rejected': False}).count()

    # How many new images do we need to obtain?
    if num_images - collection_size < 1:
        print("No new photos need to be fetched.")
        return

    group_id = styles[style][0]
    params = {
        'api_key': vislab.config['api_key'],
        'group_id': group_id,
        'per_page': 500,  # 500 is the maximum allowed
        'content_type': 1,  # only photos
    }

    # The per_page parameter is not really respected by Flickr.
    # Most of the time, we get less results than promised.
    # Accordingly, we need to make a dynamic number of requests.
    print("Old collection size: {}".format(collection_size))

    page = 0
    num_pages = np.inf
    while collection_size < num_images and page < num_pages:
        page += 1
        params['page'] = page
        url = 'http://api.flickr.com/services/rest/' + \
              '?method=flickr.photos.search&format=json&nojsoncallback=1' + \
              '&api_key={api_key}&content_type={content_type}' + \
              '&group_id={group_id}&page={page}&per_page={per_page}'
        url = url.format(**params)
        print(url)

        # Make the request and ensure it succeeds.
        page_data = json.load(urllib2.urlopen(url))
        if page_data['stat'] != 'ok':
            raise Exception("Something is wrong: API returned {}".format(
                page_data['stat']))

        # Insert the photos into the database if they are not already in it.
        photos = []
        for photo in page_data['photos']['photo']:
            image_id = 'f_' + photo['id']
            if collection.find({'image_id': image_id}).limit(1).count() == 0:
                photo['rejected'] = False
                photo['image_id'] = image_id
                photos.append(photo)
        if len(photos) > 0:
            collection.insert(photos[:(num_images - collection_size)])

        # Update collection size
        collection_size = collection.find({'rejected': False}).count()
        print("New collection size: {}".format(collection_size))

        # If there are less total results than we need, quit at this point.
        if page_data['photos']['total'] < num_images:
            break

        if page > page_data['photos']['pages']:
            print("Not enough pages to fill up to desired amount!")
            break


def populate_database(photos_per_style=1000):
    """
    Run this to collect data.
    """
    for style in styles:
        vislab.datasets.get_photos_for_style(style, photos_per_style)
