"""
Copyright Sergey Karayev / Adobe - 2013.
Written during internship at Adobe CTL, San Francisco.

Contributors:
- Helen Han - 2014 (get_tags_and_desc).

Make dataset of images with style labels by querying Flickr Groups using
the Flickr API.

Consult notebooks/flickr_dataset.ipynb for usage examples.
"""
import urllib2
import pandas as pd
import json
import vislab
import vislab.dataset
import vislab.util
import vislab.utils.distributed2

# Mapping of style names to group ids.
styles = {
    'Bokeh': ['1543486@N25'],
    'Bright': ['799643@N24'],
    'Depth of Field': ['75418467@N00', '407825@N20'],
    'Detailed': ['1670588@N24', '1131378@N23'],
    'Ethereal': ['907784@N22'],
    'Geometric Composition': ['46353124@N00'],
    'Hazy': ['38694591@N00'],
    'HDR': ['99275357@N00'],
    'Horror': ['29561404@N00'],
    'Long Exposure': ['52240257802@N01'],
    'Macro': ['52241335207@N01'],
    'Melancholy': ['70495179@N00'],
    'Minimal': ['42097308@N00'],
    'Noir': ['42109523@N00'],
    'Romantic': ['54284561@N00'],
    'Serene': ['1081625@N25'],
    'Pastel': ['1055565@N24', '1371818@N25'],
    'Sunny': ['1242213@N23'],
    'Texture': ['70176273@N00'],
    'Vintage': ['1222306@N25', "1176551@N24"],
}
style_names = styles.keys()
underscored_style_names = [
    'style_' + style.replace(' ', '_') for style in styles.keys()]


def get_df(force=False):
    """
    Load the main Flickr Style DataFrame (hits database if not cached).
    Assign train/test split to the data, making sure that items with
    more than one label end up in the test split.
    """
    filename = vislab.config['paths']['shared_data'] + '/flickr_df_mar2014.h5'
    df = vislab.util.load_or_generate_df(
        filename, _fetch_df, force)
    df['_split'] = vislab.dataset.get_train_test_split(
        df[underscored_style_names])
    df['_split'][df[underscored_style_names].sum(1) > 1] = 'test'
    return df


def get_tags_and_desc_worker(image_id):
    import flickr_api
    import vislab
    flickr_api.API_KEY = vislab.config['api_keys']['flickr']
    data = flickr_api._doget('flickr.photos.getInfo', photo_id=image_id)
    photo = data.rsp.photo
    tags = []
    if 'tag' in dir(photo.tags):
        if '__len__' in dir(photo.tags.tag):
            for i in range(0, len(photo.tags.tag)):
                tags.append(photo.tags.tag[i].raw)
        elif 'text' in dir(photo.tags.tag):
            tags = [photo.tags.tag.raw]
    return {
        'id': image_id,
        'tags': tags,
        'description': data.rsp.photo.description.text
    }


def get_tags_and_desc_manager(df):
    """
    For a given dataset, use the API to get the photo tags and
    description for each image. Store to MongoDB during the process,
    then form DataFrame to return.
    """
    job_info = {
        'module': 'vislab.datasets.flickr',
        'worker_fn': 'get_tags_and_desc_worker',
        'db_name': 'flickr',
        'collection_name': 'desc_and_tags'
    }

    query_list = df.index.tolist()
    kwargs_list = [{'image_id': _} for _ in df.index.tolist()]
    vislab.utils.distributed2.submit_to_rq(
        query_list, kwargs_list, job_info, 'get_tags_and_desc', 20)


def _get_image_url(photo, size_flag=''):
    """
    size_flag: string ['']
        See http://www.flickr.com/services/api/misc.urls.html for options.
            '': 500 px on longest side
            '_m': 240px on longest side
    """
    url = "http://farm{farm}.staticflickr.com/{server}/{id}_{secret}{size}.jpg"
    return url.format(size=size_flag, **photo)


def _get_page_url(photo):
    url = "http://www.flickr.com/photos/{owner}/{id}"
    return url.format(**photo)


def _fetch_df():
    """
    Load data from the database into a DataFrame, dropping some of the
    fetched information in favor of assembling image_urls directly.
    """
    client = vislab.util.get_mongodb_client()
    dfs = []
    for style in style_names:
        df = pd.DataFrame(list(client['flickr'][style].find()))
        df2 = pd.DataFrame(data={
            'image_url': df.apply(lambda row: _get_image_url(row), axis=1),
            'page_url': df.apply(lambda row: _get_page_url(row), axis=1)
        })
        df2['owner'] = df['owner']
        df2.index = df['image_id'].astype(str)
        style_str = 'style_' + style.replace(' ', '_')
        df2[style_str] = True
        dfs.append(df2)
    main_df = pd.concat(dfs)
    main_df = main_df.fillna(False)
    main_df[underscored_style_names] = \
        main_df[underscored_style_names].astype(bool)
    return main_df


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

    params = {
        'api_key': vislab.config['api_keys']['flickr'],
        'per_page': 500,  # 500 is the maximum allowed
        'content_type': 1,  # only photos
    }

    # The per_page parameter is not really respected by Flickr.
    # Most of the time, we get less results than promised.
    # Accordingly, we need to make a dynamic number of requests.
    print("Old collection size: {}".format(collection_size))

    page = 0
    groups = styles[style]
    # The Flickr API returns the exact same results for all pages after
    # page 9. Therefore, we set num_pages before we enter the loop.
    num_pages = num_images / params['per_page'] + 1
    while collection_size < num_images and page < num_pages:
        params['page'] = page
        page += 1

        for group in groups:
            params['group_id'] = group

            url = ('http://api.flickr.com/services/rest/' +
                   '?method=flickr.photos.search&' +
                   'format=json&nojsoncallback=1' +
                   '&api_key={api_key}&content_type={content_type}' +
                   '&group_id={group_id}&page={page}&per_page={per_page}')
            url = url.format(**params)
            print(url)

            # Make the request and ensure it succeeds.
            page_data = json.load(urllib2.urlopen(url))
            if page_data['stat'] != 'ok':
                raise Exception("Something is wrong: API returned {}".format(
                    page_data['stat']))

            # Insert the photos into the database if needed.
            photos = []
            for photo in page_data['photos']['photo']:
                if vislab.util.zero_results(
                        collection, {'image_id': photo['id']}):
                    photo['rejected'] = False
                    photo['image_id'] = photo['id']
                    photos.append(photo)
            if len(photos) > 0:
                collection.insert(photos)

            # Update collection size
            collection_size = collection.find({'rejected': False}).count()
            print("New collection size: {}".format(collection_size))


def populate_database(photos_per_style=100):
    """
    Run this to collect data into the database.
    """
    for style in style_names:
        get_photos_for_style(style, photos_per_style)


if __name__ == '__main__':
    # populate_database(photos_per_style=5000)

    # Query the API for tags and descriptions.
    df = get_df()
    df = df[df['_split'] == 'test']
    get_tags_and_desc_manager(df)
