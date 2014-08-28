"""
A searchable collection of images, listening for jobs on a Redis queue.
"""
import sys
import time
import cPickle
import sklearn
import operator
import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as metrics

import vislab
import vislab.datasets
import vislab.utils.redis_q


feats_dir = vislab.config['paths']['feats']
experiment_dir = vislab.config['paths']['shared_data'] + '/results_mar23'
feat_filenames = {
    'flickr': {
        'caffe fc6': feats_dir + '/flickr/caffe_fc6.h5',
        'caffe fc7': feats_dir + '/flickr/caffe_fc7.h5',
    }
}
dataset_loaders = {
    'flickr': vislab.datasets.flickr.get_df
}


class SearchableCollection(object):
    """
    A searchable collection of images for a given dataset.
    """
    def __init__(self, dataset_name):
        assert(dataset_name in dataset_loaders)
        t = time.time()

        # TODO: formalize
        # Load Flickr weights
        weights = cPickle.load(open('data/shared/flickr_finetune_weights.pickle'))

        # Load image information in a dataframe.
        self.images = dataset_loaders[dataset_name]()

        # Downsample [optional for dev].
        # self.images = self.images.iloc[:10000]

        # Cache the image index.
        self.index = self.images.index.tolist()
        # print self.index

        # Load all features.
        self.features = {}
        # self.S = {}
        self.features_norm = {}
        self.features_index = {}
        self.features_proj = {}
        for feature_name, filename in feat_filenames[dataset_name].iteritems():
            try:
                feats = pd.read_hdf(filename, 'df')
            except:
                feats = pd.read_pickle(filename)
            feats = feats.ix[self.images.index].values
            self.features[feature_name] = feats
            self.features_norm[feature_name] = np.sqrt(np.power(feats, 2).sum(1))

            W = weights[feature_name].T
            Wm = W - W.mean(1)[:, np.newaxis]
            self.features_proj[feature_name] = np.dot(feats, Wm)

        # Append predictions to the images DataFrame.
        if 'style scores' in self.features:
            self.images = self.images.join(
                self.features['style scores'], rsuffix='preds')

        print('Initialized SearchableCollection in {:.3f} s'.format(
            time.time() - t))

    def nn_by_id_many_filters(
            self, image_id, feature, distance,
            page=1, filter_conditions_list=None, results_per_page=8):
        """
        Return several sets of results, each filtered by different
        filter_conditions.
        """
        print('nn_by_id_many_filters', feature)
        assert(feature in self.features)
        t = time.time()

        # Get all distances to query
        nn_ind, nn_dist = self._nn(image_id, feature, distance)
        nn_df = pd.DataFrame(
            {'distance': nn_dist[1:]},
            self.images.index[nn_ind[1:]])
        images_nn_df = nn_df.join(self.images)

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

    def _nn(self, image_id, feature, distance='cosine', K=-1):
        """
        Exact nearest neighbor seach through exhaustive comparison.
        """
        # S = self.S[feature]
        feats = self.features[feature]
        feat = feats[self.index.index(image_id)]

        if distance == 'manhattan':
            dists = metrics.manhattan_distances(feat, feats)

        elif distance == 'euclidean':
            dists = metrics.euclidean_distances(feat, feats, squared=True)

        elif distance == 'chi_square':
            dists = -metrics.additive_chi2_kernel(feat, feats)

        elif distance == 'dot':
            dists = -np.dot(feats, feat)

        elif distance == 'cosine':
            feats_norm = self.features_norm[feature]
            dists = -np.dot(feats, feat) / feats_norm / np.linalg.norm(feat, 2)

        elif distance == 'projected':
            feats = self.features_proj[feature]
            feat = feats[self.index.index(image_id)]
            dists = sklearn.utils.extmath.row_norms(feats - feat)

        dists = dists.flatten()
        if K > 0:
            nn_ind = np.argsort(dists).flatten()[:K]
        else:
            nn_ind = np.argsort(dists)
        nn_dist = dists[nn_ind]

        return nn_ind, nn_dist


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


def run_worker(dataset_name='flickr'):
    """
    Initialize a searchable collection and start listening for jobs
    on a redis queue.
    """
    collection = SearchableCollection(dataset_name)
    registered_functions = {
        'nn_by_id_many_filters': collection.nn_by_id_many_filters,
    }
    vislab.utils.redis_q.poll_for_jobs(
        registered_functions, 'similarity_server')


if __name__ == '__main__':
    print("usage: python searchable_collection.py <dataset_name>")
    assert(len(sys.argv) == 2)
    dataset_name = sys.argv[1]
    run_worker(dataset_name)
