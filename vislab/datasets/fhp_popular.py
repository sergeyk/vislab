import pymongo
import time
import requests
from requests_oauthlib import OAuth1
import vislab


auth = OAuth1(*vislab.config['api_keys']['500px'])
popular_url = 'https://api.500px.com/v1/photos?feature=popular'
likes_url = 'https://api.500px.com/v1/photos/{}/votes'
favs_url = 'https://api.500px.com/v1/photos/{}/favorites'
user_photos_url = 'https://api.500px.com/v1/photos?feature=user_favorites&user_id={}'


def process_all_pages(
        url, process, key, params={'rpp': 100}, page_limit=float('inf')):
    """
    Return list of info. Get all results for a request by going through
    all pages.

    Parameters
    -----------
    page_limit: limit number of pages to process. default is no limit
    url: HTTP GET request url
    process: lambda function, returns a list
    key: indicates which key of the json to process
    params: dict to pass parameters in url
    page_limit: limit number of pages to process. default is no limit
    """
    params['page'] = 1
    retries = 0
    while retries < 3:
        try:
            r = requests.get(url, params=params, auth=auth)
            if r.status_code == 200:
                total_pages = r.json()['total_pages']
                info = []
                while (params['page'] <= total_pages and
                       params['page'] <= page_limit):
                    #print "Now on page {} for {}".format(params['page'], url)
                    r = requests.get(url, params=params, auth=auth)
                    if r.status_code == 200:
                        info += process(r.json()[key])
                        params['page'] += 1
                return info
            else:
                print "Error, status is {}".format(r.status_code)
                print "The url is {}".format(url)
                return []
        except Exception as e:
            print e
            print "On retry {}".format(retries)
            retries += 1
            time.sleep(5)


def insert_pop_ids(col):
    """
    Get a list of the (photo id, photographer id) of the popular photos
    Insert list into col.
    """
    process = lambda photos: [(p['id'], p['user']['id']) for p in photos]
    info = process_all_pages(popular_url, process, 'photos')
    print "Finished getting list of ids. Now inserting into db..."
    for i in info:
        col.insert({'photo_id': i[0],
                    'user_id': i[1]})
    print "Finished inserting into db"


def get_info(id):
    """
    Given the id of a popular photo, get the info of users who
    like/favorite it.
    """
    process = lambda users: [user['id'] for user in users]
    likes = process_all_pages(likes_url.format(id), process, 'users')
    print "Got {} likes from photo {}".format(len(likes), id)
    favs = process_all_pages(favs_url.format(id), process, 'users')
    print "Got {} favs from photo {}".format(len(favs), id)
    return likes, favs


def insert_photo_info(ids_col, info_col):
    """
    Given a collection of ids,
    insert photo info into info collection
    """
    cursor = ids_col.find(timeout=False)
    for entry in cursor:
        photo_id = entry['photo_id']
        user_id = entry['user_id']
        if info_col.find({'photo_id': photo_id}).limit(1).count() == 0:
            likes, favs = get_info(photo_id)
            photo_data = {
                'photo_id': photo_id,
                'user_id': user_id,
                'likes': likes,
                'favs': favs
             }
            info_col.insert(photo_data)
        else:
            print "Already inserted photo{}".format(entry['photo_id'])


def insert_photo_info_from_ids(ids, info_col, source, count):
    """
    Given a list of (photo_id, user_id)
    insert photo info into info collection
    """
    for (photo_id, user_id) in ids:
        if info_col.find({'photo_id': photo_id}).limit(1).count() == 0:
            likes, favs = get_info(photo_id)
            photo_data = {
                'photo_id': photo_id,
                'user_id': user_id,
                'likes': likes,
                'favs': favs
             }
            info_col.insert(photo_data)
            print "Inserted photo {} found from user  {}. Cursor at {}".format(
                photo_id, source, count)
        else:
            print "Already inserted photo{}".format(photo_id)


def find_other_photos(ids_col, info_col):
    """
    NOTE: api is weird and photos?user={} does not return photos of a user!!!!
    Given collections of popular user and photo ids,
    find the ids of a popular user's favorites and get info.
    """
    cursor = ids_col.find(timeout=False).sort('user_id', pymongo.DESCENDING)
    cursor = cursor.skip(1385)
    count = 1385
    #last user was 7484777
    for entry in cursor:
        count += 1
        user_id = entry['user_id']
        url = user_photos_url.format(user_id)
        process = lambda photos: [(p['id'], p['user_id']) for p in photos]
        ids = process_all_pages(url, process, 'photos', page_limit=100)
        insert_photo_info_from_ids(ids, info_col, user_id, count)
        print "-----Done inserting favorites from {}----------".format(user_id)


if __name__ == '__main__':
    client = pymongo.MongoClient('localhost', 27017)
    db = client['500px']
    popular_col = db['popular_info']
    ids_col = db['ids']
    #insert_pop_ids(ids_col)
    #insert_photo_info(ids_col, popular_col)
    find_other_photos(ids_col, popular_col)
