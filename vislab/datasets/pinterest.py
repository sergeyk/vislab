"""
Copyright Helen Han 2013.

Script to scrape Pinterest board, pin, and user information.
Takes command-line argument for which of these to do.
Stores results in database.

We got a *lot* of data: 1.3M pins for about 40 queries, 20 of which
come from the Flickr style dataset.
"""
# coding=utf-8
#!/usr/bin/python
import time
import numpy as np
import pandas as pd
import sys
import bs4
import pymongo
from selenium import webdriver
import traceback
import vislab
import vislab.datasets.flickr


def get_pins_80k_df(force=False):
    """
    This dataset is exactly the same as the Flickr dataset, except using
    Pinterest board queries to fetch the results.
    """
    underscored_style_names = vislab.datasets.flickr.underscored_style_names
    filename = vislab.config['paths']['shared_data'] + \
        '/pins_df_80k_mar2014.h5'
    df = vislab.util.load_or_generate_df(
        filename, _fetch_pins_80k_df, force)
    df['_split'] = vislab.dataset.get_train_test_split(
        df[underscored_style_names])
    df['_split'][df[underscored_style_names].sum(1) > 1] = 'test'

    return df


def _fetch_pins_80k_df():
    pins_df = pd.read_hdf('../data/unshared/pins_df_mar21.h5', 'df')
    pins_df.columns = [_.lower() for _ in pins_df.columns]
    query_names = [
        'query_' + style.lower()
        for style in vislab.datasets.flickr.style_names
    ]

    indices = []
    for query_name in query_names:
        if not query_name in pins_df.columns:
            print style
        ind = pins_df[query_name]
        indices += pins_df.index[ind][
            np.random.permutation(ind.sum())[:4000]].tolist()
    assert len(indices) == len(np.unique(indices))

    pins_df_small = pins_df.ix[indices][query_names]
    # should be all 4000:
    print pins_df_small[query_names].sum(0)
    # should be only 1:
    print pins_df_small[query_names].sum(1).value_counts()

    # Rename to match Flickr's names
    assert np.all(pins_df_small.columns == query_names)
    pins_df_small.columns = vislab.datasets.flickr.underscored_style_names

    pins_df_small['image_url'] = pins_df['img']
    pins_df_small['page_url'] = [
        'http://pinterest.com/pin/{}/'.format(_) for _ in pins_df_small.index]
    return pins_df_small


def get_driver(driver_width=6000, driver_height=3000, limit=3):
    connections_attempted = 0
    while connections_attempted < limit:
        try:
            driver = webdriver.PhantomJS(service_args=['--load-images=no'])
            driver.set_window_size(driver_width, driver_height)
            return driver
        except Exception as e:
            connections_attempted += 1
            print('Getting driver again...')
            print('  connections attempted: {}'.format(connections_attempted))
            print('  exception message: {}'.format(e))
            traceback.print_exc()


# for any results page, the web driver scrolls down until the number of
# results reaches the limit (limit is 500 or 1000 depending on time of day...)
#for now, get users for a query
def process_whole_page(driver, url, process, limit=500,
                       connections_to_attempt=3, scrolls_to_attempt=3,
                       sleep_interval=2):
    """
    Process the whole page at url with the given function, making sure
    that at least limit results have been processed -- or that there
    are less than limit results on the page.
    To do this, we scroll down the page with driver.

    Parameters
    ----------
    driver: selenium.webdriver
    url: string
    process: function
        Text fetched by driver is processed by this.
        Returns a list.
    limit: int
        Until we get this many results, or become certain that there
        aren't this many results at the url, we will keep scrolling the
        driver.
    connections_to_attempt: int
    scrolls_to_attempt: int
    sleep_interval: float
        Sleep this number of seconds between tries.

    Returns
    -------
    results: list or None
        Whatever the process function returns.

    Raises
    ------
    e: Exception
        If connection times out more than connections_to_attempt
    """
    assert(scrolls_to_attempt > 0)
    assert(limit > 0)
    driver = get_driver()
    connections_attempted = 0
    while connections_attempted < connections_to_attempt:
        try:
            driver.get(url)
            soup = bs4.BeautifulSoup(driver.page_source)
            results = process(soup)
            all_scrolls_attempted = 0

            # If we fetch more than limit results already, we're done.
            # Otherwise, try to get more results by scrolling.
            # We give up after some number of scroll tries.
            # If we do get more results, then the scroll count resets.
            if len(results) < limit:
                scrolls_attempted = 0
                while (scrolls_attempted < scrolls_to_attempt and
                       len(results) < limit):
                    all_scrolls_attempted += 1
                    scrolls_attempted += 1

                    # Scroll and parse results again.
                    # The old results are still on the page, so it's fine
                    # to overwrite.
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);")
                    soup = bs4.BeautifulSoup(driver.page_source)
                    new_results = process(soup)

                    if len(new_results) > len(results):
                        results = new_results
                        scrolls_attempted = 0

            print('Obtained {} results after {} scrolls'.format(
                len(results), all_scrolls_attempted))
            if len(results) > limit:
                results = results[:limit]
            return results

        except Exception as e:
            connections_attempted += 1
            print('URL failed: {}'.format(url))
            print('  connections attempted: {}'.format(connections_attempted))
            print('  exception message: {}'.format(e))
            traceback.print_exc()
            time.sleep(sleep_interval)
            driver = get_driver()

    print('URL skipped: {}'.format(url))
    return None


def parse_board_object(board, username, source):
    """
    Parameters
    ----------
    board: some kind of bs4 object

    Returns
    -------
    data: dict
    """
    num_pins = board.select('.boardPinCount')[0].text.strip()
    num_pins = num_pins.split(' ')[0].replace(',', '')
    data = {
        'username': username,
        'source': source,
        'board_url': 'www.pinterest.com{}'.format(
            board.find_all('a', {'class': 'boardLinkWrapper'})[0].get('href')),
        'board_name': board.select('.boardName .title')[0].text.strip(),
        'cover_img': board.select('img')[0]['src'],
        'thumb_imgs': [_['src'] for _ in board.select('img')[1:]],
        'num_pins': int(num_pins)
    }
    data['_id'] = data['board_url']
    return data


def scrape_user_boards(driver, username, source):
    """
    Return all boards of the user.

    Parameters
    ----------
    username: string
    source: string
        Describes why this user's boards are getting scraped:
        - "query: <whatever>"
        - "follower of: <username>"

    Returns
    -------
    boards: list of dicts
    """
    url = 'http://www.pinterest.com/{}/boards/'.format(username)
    boards = process_whole_page(
        driver, url, lambda soup: [
            parse_board_object(board, username, source)
            for board in soup.select('div.item')
        ])
    return boards


def get_followers(db):
    """
    Iterate through users in db and get usernames of followers and following
    """
    driver = get_driver()
    user_iterator = db.users.find(fields={'query': False, 'num_boards': False})
    for user in user_iterator:
        username = user['username']
        url = 'http://www.pinterest.com/{}'.format(username)
        followers = get_followers_list(url, driver)
        following = get_followers_list(url, driver, False)
        # Note: after update, index changes  for next time
        db.users.update(
            {'username': username},
            {'$set': {'followers': followers, 'following': following}}
        )
        print('Updated {}. Followers: {}  Following: {}'.format(
            username, len(followers), len(following)))


def get_followers_list(user_url, driver, followers=True):
    """
    Returns a list of users who follow or are followed by a user.

    Parameters
    ----------
    user_url: string
    driver: selenium.webdriver
    followers: bool
        If True, gets users who are followers of this user.
        If False, gets users who this user follows.
    """
    if followers:
        url = user_url + '/followers/'
    else:
        url = user_url + '/following/'
    process = lambda soup: [
        str(item.find_all('a',
            {'class': 'userWrapper'})[0].get('href'))
        for item in soup.select('div.item')
    ]

    followers = process_whole_page(driver, url, process)
    return followers


def get_usernames_from_query_results_page(driver, url, limit=500):
    # Parse the usernames, which exist as the last part of hrefs in
    # .boardLinkWrapper <a> tags.
    get_usernames = lambda soup: [
        link['href'].split('/')[1]
        for link in soup.findAll('a', {'class': 'boardLinkWrapper'})
    ]
    usernames = process_whole_page(driver, url, get_usernames, limit)
    return usernames


def scrape_boards(query, board_collection, user_collection, user_limit=500):
    """
    Find all users with a board matching the query, and scrape all of
    their boards, inserting them into board_collection.

    If user is already present in user_collection, it means that we
    have scraped their boards, and so do not need to scrape again -- but
    do need to update their record with this query.
    """
    print(u'scrape_boards: on {}'.format(query))

    driver = get_driver()
    url = 'http://www.pinterest.com/search/boards/?q=' + query
    usernames = get_usernames_from_query_results_page(driver, url, user_limit)

    # For each user, scrape their boards page, or update their record.
    for username in usernames:
        # If we've already scraped this user's boards, then we don't
        # need to do that again, but we note that we have seen this
        # user for this new query.
        username_count = user_collection.find(
            {'username': username}).limit(1).count()
        if username_count > 0:
            # TODO: confirm that this appends to list of queries
            # because it looks like it just overwrites it
            if query in user_collection.find_one(
                    {'username': username})['query']:
                print('Already ran query {} for user {}'.format(
                    query, username))
            else:
                print("Already ran a different query for user {}".format(
                    username))
                user_collection.update(
                    {'username': username}, {'$push': {'query': query}})

        # Otherwise, we scrape the user's boards.
        else:
            boards = scrape_user_boards(
                driver, username, 'query: {} '.format(query))
            if len(boards) > 0:
                user_collection.insert({
                    'username': username,
                    'num_boards': len(boards),
                    'query': [query]
                })
                for board in boards:
                    try:
                        board_collection.insert(board)
                    except pymongo.errors.DuplicateKeyError:
                        continue
                print('Inserted {} from {} with query: {}'.format(
                    len(boards), username, query))


def get_boards(driver, query):
    """
    Given a query, return the list of boards dicts that show up.
    """
    url = 'http://www.pinterest.com/search/boards/?q=' + query
    boards = process_whole_page(
        driver, url, lambda soup: [
            parse_board_page(board, query)
            for board in soup.select('div.item')
        ])
    return boards


def get_boards_for_queries(db, queries):
    user_collection = db['users']
    user_collection.ensure_index('username')
    board_collection = db['boards']
    for query in queries:
        query = query.lower()
        scrape_boards(query, board_collection, user_collection)


#source: url it was pinned from
#img: largest resolution is 736x for all images it seems
def parse_pin(pin, username, board_name, query):
    #from IPython import embed
    #embed()
    pinned_from = pin.select('h4.pinDomain')
    repins = pin.select('em.socialMetaCount')
    if len(repins) == 0:
        repins = 0
    else:
        repins = int(repins[0].text.strip())
    if len(pinned_from) == 0:
        pinned_from = username
    else:
        pinned_from = pinned_from[0].text.strip()
    caption = pin.select('img')[0]
    try:
        caption = caption['alt']
    except Exception:
        caption = ''
    data = {
        'username': username,
        'pin_url': 'www.pinterest.com{}'.format(
            pin.find('a', {'class': 'pinImageWrapper'}).get('href')),
        'repins_likes_url': ['www.pinterest.com{}'.format(
            link['href']) for link in pin.select('a.socialItem')],
        'caption': caption,
        'source': pinned_from,
        'img': pin.select('img')[0]['src'].replace('236x', '736x'),
        'repins': repins,
        'board_name': board_name,
        'query': [query]
    }
    data['_id'] = data['pin_url']
    return data


def parse_board_page(board, query):
    url = board.find_all('a', {'class': 'boardLinkWrapper'})[0].get('href')
    data = {
        'username': url.split('/')[1],
        'board_url': 'http://www.pinterest.com{}'.format(url),
        'board_name': url.split('/')[2],
        'query': query
    }
    return data


def scrape_pins(driver, board, pin_collection):
    url = board['board_url']
    driver.get(url)

    pins = process_whole_page(
        driver, url, lambda soup: [parse_pin(
            pin, board['username'], board['board_name'], board['query'])
            for pin in soup.select('div.item')
        ])
    for pin in pins:
        if pin_collection.find({'_id': pin['_id']}).count() == 0:
            pin_collection.insert(pin)
            print('Inserted {} from board {}: {}'.format(
                pin['pin_url'], board['board_url'], board['query']))
        else:
            print('{} has already been added'.format(pin['pin_url']))
            pin_collection.update(
                {'pin_url': pin['pin_url']},
                {'$push': {'query': board['query']}}
            )


def get_pins_for_queries(db, queries):
    pin_collection = db['pins']
    driver = get_driver()
    for query in queries:
        board_iterator = get_boards(driver, query)
        for board in board_iterator:
            doc = {
                'board_name': board['board_name'],
                'username': board['username']
            }
            if vislab.util.zero_results(pin_collection, doc):
                print("Scraping: {}".format(board['board_url']))
                t = time.time()
                scrape_pins(driver, board, pin_collection)
                print("...took {:.2f} s".format(time.time() - t))
            else:
                print("Not scraping {}".format(doc))
        print('Done scraping pins for {}'.format(query))


if __name__ == '__main__':
    print("usage: scrape.py <mode>")
    assert(len(sys.argv) == 2)
    mode = sys.argv[1]

    queries = vislab.datasets.flickr.style_names + [
        'energetic', 'washed out', 'instagram', 'sepia', 'black and white',
        'corporate', 'industrial', 'organic', 'nature', 'portrait',
        'landscape', 'animals', 'happy', 'scary', 'sad', 'calm', 'upbeat',
        'pensive', 'tense', 'futuristic', 'sleek', 'radiant', 'fall',
        'summer', 'winter', 'spring', 'cloudy', 'night'
    ]

    client = vislab.util.get_mongodb_client()
    db = client['pinscraping']

    if mode == 'followers':
        get_followers(db)

    elif mode == 'boards':
        get_boards_for_queries(db, queries)

    elif mode == 'pins':
        get_pins_for_queries(db, queries)

    else:
        raise ValueError("Invalid mode")
