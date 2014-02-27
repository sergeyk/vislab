"""
Scraping the dpchallenge.net website for user-photo favorite relationships.
Can start with the AVA dataset.
"""
import requests
import bs4


def fav_user_ids(image_ids):
    ids_ = []
    feats = []
    for image_id in image_ids:
        try:
            url = 'http://www.dpchallenge.com/favorites.php?IMAGE_ID={}'
            r = requests.get(url.format(image_id))
            soup = bs4.BeautifulSoup(r.text)
            tags = soup.findAll(
                lambda tag: 'href' in tag.attrs and
                tag.attrs['href'].startswith('profile.php?USER_ID=')
                and tag.parent.name == 'td')
            user_ids = [tag.attrs['href'].split('=')[1] for tag in tags]
            added_dates = [
                list(tag.parent.parent.children)[3].text.strip()
                for tag in tags
            ]

            ids_.append(image_id)
            feats.append(zip(user_ids, added_dates))
        except Exception as e:
            print(e)
    return ids_, feats
