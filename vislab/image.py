import os
import requests
import skimage.io
import vislab
import vislab.datasets


def fetch_image_filename_for_id(image_id, dataset_name, force=False):
    """
    Fetch the full-size image for image id from the web, and save into dirname.
    For example, (id='359334', dirname='temp') will fetch image from
    http://www.dpchallenge.com/image.php?IMAGE_ID=359334 and save it
    into 'temp/359334.jpg'.

    Handles both AVA and Flickr images.

    Parameters
    ----------
    image_id: int or string
    dataset_name: string
    force: boolean [False]

    Returns
    -------
    filename: string
    """
    # PASCAL does not have image urls, all on HD.
    if dataset_name == 'pascal':
        filename = vislab.datasets.pascal.get_image_filename_for_id(image_id)

    # Other datasets have images on the internet that need to be fetched
    else:
        filename = vislab.config['images'] + '/{}.jpg'.format(image_id)
        if force or not os.path.exists(filename):
            try:
                if dataset_name == 'flickr':
                    url_fn = vislab.datasets.flickr.get_image_url_for_id
                elif dataset_name == 'wikipaintings':
                    url_fn = vislab.datasets.wikipaintings.get_image_url_for_id
                elif dataset_name in ['ava', 'ava_style']:
                    url_fn = vislab.datasets.ava.get_image_url_for_id
                elif dataset_name in ['behance_photo']:
                    url_fn = vislab.datasets.behance.get_image_url_for_photo_id
                elif dataset_name in ['behance_illustration']:
                    url_fn = vislab.datasets.behance.get_image_url_for_illustration_id
                else:
                    raise ValueError("Uknown dataset_name")

                print("Downloading image for id: {}".format(image_id))
                img_url = url_fn(image_id)
                r = requests.get(img_url)
                with open(filename, 'w') as f:
                    f.write(r.content)
            except ValueError as e:
                raise e
            except Exception as e:
                print("Exception: {}".format(e))
                return None
    return filename


def get_image_for_filename(image_filename):
    if image_filename is not None:
        image = skimage.io.imread(image_filename)
        return image
    else:
        return None


def get_image_for_id(image_id, dataset_name, force=False):
    """
    Handles both AVA and Flickr images.
    """
    image_filename = fetch_image_filename_for_id(image_id, dataset_name)
    return get_image_for_filename(image_filename)
