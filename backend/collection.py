"""
A searchable collection of images, listening for jobs on a Redis queue.
"""
import operator
import os
import bottleneck as bn
import pandas as pd
import numpy as np
import time
import sklearn.metrics.pairwise as metrics
import aphrodite.flickr
from vislab.backend import redis_q


class Collection(object):
    """
    A searchable collection of images.
    """
    def __init__(self):
        # Load image information in a dataframe.
        self.images = aphrodite.flickr.load_flickr_df()

        # Downsample if needed.
        self.images = self.images.iloc[:10000]

        # Load predictions, which are used to filter results and as a feature.
        preds_filename = os.path.expanduser(
            '~/work/aphrodite/data/results/flickr_decaf_fc6_preds.h5')
        preds = pd.read_hdf(preds_filename, 'df')
        preds = preds.ix[self.images.index]
        self.images = self.images.join(preds)

        # Load all features.
        feats_filename = os.path.expanduser(
            '~/work/aphrodite/data/feats/flickr/decaf_fc6_arr_df.h5')
        feats = pd.read_hdf(feats_filename, 'df')
        feats = feats.ix[self.images.index]

        self.features = {
            'deep learned fc6': feats,
            'style scores': preds
        }

    def get_random_id(self):
        """
        Return random image id from the collection.
        """
        ind = np.random.randint(self.images.shape[0] + 1)
        return self.images.index[ind]

    def find_by_id(self, id_):
        """
        Return dict of everything we know about the image at id.
        """
        image = self.images.ix[id_]
        return image.to_dict()

    def nn_by_id_many_filters(self, image_id, feature, distance, page=1,
                              filter_conditions_list=None, results_per_page=8):
        """
        Return several sets of results, each filtered by different
        filter_conditions.
        """
        assert(feature in self.features)
        t = time.time()

        # Get all distances to query
        images = self.images
        feats = self.features[feature]
        feat = feats.ix[image_id].values
        feats = feats.ix[images.index].values

        nn_ind, nn_dist = nn(feat, feats, distance)
        nn_df = pd.DataFrame(
            {'distance': nn_dist[1:]},
            images.index[nn_ind[1:]])
        images_nn_df = nn_df.join(images)

        # TODO: figure out what to do for paging
        start_ind = 0
        end_ind = start_ind + results_per_page

        results_sets = []
        for conditions in filter_conditions_list:
            result_images = filter_df(images_nn_df, conditions)
            num_results = result_images.shape[0]
            result_images = result_images[start_ind:end_ind]
            result_images['image_id'] = result_images.index
            results = [row.to_dict() for _, row in result_images.iterrows()]
            results_data = {
                'results': results,
                'start_ind': start_ind,
                'page': page,
                'num_results': num_results,
                'time_elapsed': time.time() - t
            }
            results_sets.append(results_data)
        return results_sets

    def nn_by_id(self, image_id, feature, distance, page=1,
                 filter_conditions=None, results_per_page=32):
        """
        Fetch nearest neighbors for image at given id.
        """
        assert(feature in self.features)
        t = time.time()

        # Filter on images if given filter.
        images = filter_df(self.images, filter_conditions)
        num_results = images.shape[0]

        # Compute feature distances among candidates.
        feats = self.features[feature]
        feat = feats.ix[image_id].values
        feats = feats.ix[images.index].values

        # Discount the first element, because it's the query.
        K = results_per_page * page
        nn_ind, nn_dist = nn(feat, feats, distance, K + 1)
        start_ind = (page - 1) * results_per_page + 1

        result_images = images.iloc[nn_ind[start_ind:]]
        result_images['image_id'] = result_images.index
        result_images['distance'] = nn_dist[start_ind:]
        results = [row.to_dict() for _, row in result_images.iterrows()]

        return {
            'results': results,
            'start_ind': start_ind,
            'page': page,
            'num_results': num_results,
            'time_elapsed': time.time() - t
        }


def filter_df(df, conditions):
    """
    Filter the DataFrame based on free-form conditions.

    Parameters
    ----------
    df: pandas.DataFrame
    filter_conditions: dict
        column: condition as str
        ex: {'pred_HDR': '> 0'}
    """
    if conditions is None or len(conditions) == 0:
        return df
    inds = [eval("df['{}'] {}".format(col, condition))
            for col, condition in conditions.iteritems()]
    ind = reduce(operator.and_, inds)
    return df[ind]


def nn(feat, feats, distance='euclidean', K=-1):
    """
    Exact nearest neighbor seach through exhaustive comparison.
    """
    if distance == 'manhattan':
        dists = metrics.manhattan_distances(feat, feats)
    elif distance == 'euclidean':
        dists = metrics.euclidean_distances(feat, feats, squared=True)
    elif distance == 'chi_square':
        dists = -metrics.additive_chi2_kernel(feat, feats)

    dists = dists.flatten()
    if K > 0:
        nn_ind = bn.argpartsort(dists, K).flatten()[:K]
        nn_ind = nn_ind[np.argsort(dists[nn_ind])]
    else:
        nn_ind = np.argsort(dists)
    nn_dist = dists[nn_ind]

    return nn_ind, nn_dist


if __name__ == '__main__':
    collection = Collection()

    def do_job(method_name, kwargs):
        assert(method_name in dir(collection))
        results_sets = eval('collection.{}(**{})'.format(method_name, kwargs))
        print("Returning results")
        return results_sets
    redis_q.wait_for_job(do_job, 'similarity_server')
