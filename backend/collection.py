import os
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import \
    manhattan_distances, euclidean_distances, additive_chi2_kernel
import aphrodite.flickr

class Collection(object):

    def __init__(self):
        # load dataframe
        self.images = aphrodite.flickr.load_flickr_df()
        self.images = self.images.iloc[:1000]

        self.feats = pd.read_hdf(
            os.path.expanduser('~/work/aphrodite/data/feats/flickr/decaf_fc6_arr_df.h5'),
            'df')
        self.feats = self.feats.ix[self.images.index]

    def find_by_id(self, id_):
        """
        Return dict of everything we know about the image at id.
        """
        image = self.images.ix[id_]
        return image.to_dict()

    def nn_by_id(self, id_, feature, distance, page=1, style='all'):
        t = time.time()

        # Filter on images if given filter.
        images = self.images
        if style != 'all':
            images = images[images[style]]

        # Compute feature distances among candidates.
        # TODO: handle different features
        feat = self.feats.ix[id_].values
        feats = self.feats.ix[images.index].values

        # Discount the first element, because it's the query.
        results_per_page = 50
        K = results_per_page * page
        nn_ind, nn_dist = nn(feat, feats, distance, K + 1)
        start_ind = (page - 1) * results_per_page + 1

        result_images = images.iloc[nn_ind[start_ind:]]
        result_images['image_id'] = result_images.index
        result_images['distance'] = nn_dist[start_ind:]

        results = [row.to_dict() for ix, row in result_images.iterrows()]
        num_results = images.shape[0]

        return {
            'results': results,
            'start_ind': start_ind,
            'page': page,
            'num_results': num_results,
            'time_elapsed': time.time() - t
        }


def nn(feat, feats, distance='euclidean', K=50):
    """
    Exact nearest neighbor seach through exhaustive comparison.
    """
    if distance == 'manhattan':
        dists = manhattan_distances(feat, feats)
    elif distance == 'euclidean':
        dists = euclidean_distances(feat, feats, squared=True)
    elif distance == 'chi_square':
        dists = -additive_chi2_kernel(feat, feats)

    dists = dists.flatten()
    #nn_ind = bn.argpartsort(dists, K).flatten()[:K]
    nn_ind = np.argsort(dists).flatten()[:K + 1]
    nn_dist = dists[nn_ind]

    return nn_ind, nn_dist
