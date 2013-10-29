"""
Script to download and process 1024 images and write them out to file.
A job consists of a (id, image_url, label) tuple.
"""
import random
import string
import os
import requests
import aphrodite
import vislab
import cPickle
import numpy as np
import decaf.util.transform
import skimage.io


queue_name = 'finetune_convnet'
IMAGES_DIRNAME = vislab.util.makedirs(aphrodite.REPO_DIRNAME + '/data/images')
num_images = 1024


def process_image(image_id, image_url):
    filename = IMAGES_DIRNAME + '/{}.jpg'.format(image_id)
    try:
        if not os.path.exists(filename):
            print("Downloading image for id: {}".format(image_id))
            r = requests.get(image_url)
            with open(filename, 'w') as f:
                f.write(r.content)
        print("Loading image for id: {}".format(image_id))
        image = skimage.io.imread(filename)
        image = decaf.util.transform.scale_and_extract(
            decaf.util.transform.as_rgb(image), 256)
        image = (image * 255.).astype('uint8')
        flat_image = image.swapaxes(0, 1).swapaxes(0, 2).flatten()
        return flat_image
    except Exception as e:
        print("Exception: {}".format(e))
        return None


if __name__ == '__main__':
    output_dirname = vislab.util.makedirs(
        vislab.config['paths']['feats'] + '/minibatches')
    random.seed()

    # Process images until enough have been processed.
    redis_conn = vislab.util.get_redis_conn()
    data = []
    labels = []
    while len(data) < num_images:
        msg = redis_conn.blpop(queue_name, timeout=5)
        if msg is None:
            print("No more jobs on the queue, writing and exiting.")
            break
        image_id, info = cPickle.loads(msg[1])
        try:
            image = process_image(image_id, info['image_url'])
            if image is not None:
                data.append(image)
                labels.append(info['label'])
        except Exception as e:
            print('Exception: {}'.format(e))
    data = np.ascontiguousarray(np.array(data).T)
    labels = np.array(labels).astype(int)

    # Save data if full minibatch.
    if labels.shape[0] == num_images:
        N = 8
        random_name = ''.join(
            random.choice(string.ascii_uppercase + string.digits)
            for x in range(N)
        )
        filename = '{}/{}_{}'.format(output_dirname, random_name, labels.shape[0])
        np.savez_compressed(filename, data=data, labels=labels)
