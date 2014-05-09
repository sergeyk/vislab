"""
A searchable collection of images, listening for jobs on a Redis queue.
"""
import sys
import operator
import time
import bottleneck as bn
import pandas as pd
import numpy as np
import sklearn.metrics.pairwise as metrics
import vislab
import vislab.utils.redis_q
import vislab.datasets


feats_dir = vislab.config['paths']['feats']
experiment_dir = vislab.config['paths']['shared_data'] + '/results_mar23'
feat_filenames = {
    'flickr': {
        'caffe fc6': feats_dir + '/flickr/caffe_fc6.h5',
        'style scores': experiment_dir + '/flickr_mar23_preds_caffe_fc6_df.h5'
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

        # Load image information in a dataframe.
        self.images = dataset_loaders[dataset_name]()

        # Downsample [optional for dev].
        # self.images = self.images.iloc[:10000]

        # Load all features.
        self.features = {}
        for feature_name, filename in feat_filenames[dataset_name].iteritems():
            try:
                feats = pd.read_hdf(filename, 'df')
            except:
                feats = pd.read_pickle(filename)
            feats = feats.ix[self.images.index]
            self.features[feature_name] = feats

        # Append predictions to the images DataFrame.
        if 'style scores' in self.features:
            self.images = self.images.join(
                self.features['style scores'], rsuffix='preds')

        print('Initialized SearchableCollection in {:.3f} s'.format(
            time.time() - t))

    def nn_by_id_many_filters(self, image_id, feature, distance, page=1,
                              filter_conditions_list=None, results_per_page=8):
        """
        Return several sets of results, each filtered by different
        filter_conditions.
        """
        print(feature)
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
        print feature
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
    elif distance == 'dot':
        dists = -np.dot(feat, feats)

    dists = dists.flatten()
    if K > 0:
        nn_ind = bn.argpartsort(dists, K).flatten()[:K]
        nn_ind = nn_ind[np.argsort(dists[nn_ind])]
    else:
        nn_ind = np.argsort(dists)
    nn_dist = dists[nn_ind]

    return nn_ind, nn_dist


def run_worker(dataset_name='flickr'):
    """
    Initialize a searchable collection and start listening for jobs.
    """
    collection = SearchableCollection(dataset_name)
    registered_functions = {
        'nn_by_id_many_filters': collection.nn_by_id_many_filters
    }
    vislab.utils.redis_q.poll_for_jobs(
        registered_functions, 'similarity_server')


if __name__ == '__main__':
    print("usage: python searchable_collection.py <dataset_name>")
    dataset_name = sys.argv[1]
    run_worker(dataset_name)
