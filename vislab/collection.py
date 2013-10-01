"""
Interface to a Mongo database of images.
"""
import numpy as np
from vislab import util


class Collection(object):

    def __init__(self):
        db = util.get_mongodb_client()['images']
        names = set(db.collection_names()) - set(['system.indexes'])
        self.image_ids = dict([
            (name, [x['image_id']for x in db[name].find(fields=['image_id'])])
            for name in names
        ])
        self.db = db

    def get_ids_and_collection(self, name):
        assert(name in self.image_ids)
        image_ids = self.image_ids[name]
        collection = self.db[name]
        return image_ids, collection

    def get_random_id(self, name='flickr'):
        """
        Return random image id from the collection.
        """
        image_ids, _ = self.get_ids_and_collection(name)
        ind = np.random.randint(len(image_ids) + 1)
        return image_ids[ind]

    def find_by_id(self, image_id, name='flickr'):
        """
        Return dict of everything we know about the image at id.
        """
        image_ids, collection = self.get_ids_and_collection(name)
        assert(image_id in image_ids)
        return collection.find_one({'image_id': image_id})
