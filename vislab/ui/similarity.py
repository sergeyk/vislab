import numpy as np
import flask
import vislab.collection
import vislab.datasets
import vislab.utils.redis_q
import vislab.ui

app = flask.Flask(__name__)

# TODO: get this working again
# collection = vislab.collection.Collection('ui_datasets')
# collection_name = 'flickr'
df = vislab.datasets.flickr.get_df()


@app.route('/')
def index():
    return flask.redirect(flask.url_for('similar_to_random'))


@app.route('/similar_to_random')
def similar_to_random():
    # TODO: get this working again
    # image_id = collection.get_random_id(collection_name)
    image_id = df.index[np.random.randint(df.shape[0] + 1)]
    # TODO: actually, need to get random ind from searchable_collection,
    # since it might be a downsampled set
    return flask.redirect(flask.url_for('similar_to_id', image_id=image_id))


@app.route('/similar_to/<image_id>/<feature>/<distance>')
def similar_to_id(image_id, feature, distance):
    """
    This function does double duty: it returns both the rendered HTML
    and the JSON results, depending on whether the json arg is set.
    This keeps the parameter-parsing logic in one place.
    """
    select_options = {
        'feature': {
            'name': 'feature',
            'options': ['caffe fc6']
        },
        'distance': {
            'name': 'distance_metric',
            'options': ['euclidean', 'dot', 'manhattan', 'chi_square']
        }
    }

    args = vislab.ui.util.get_query_args(
        defaults={
            'feature': 'caffe fc6',
            'distance': 'euclidean',
            'page': 1,
        },
        types={
            'page': int
        }
    )

    prediction_options = ['all'] + [
        'pred_{}'.format(x)
        for x in vislab.datasets.flickr.underscored_style_names
    ]

    filter_conditions_list = []
    for prediction in prediction_options:
        filter_conditions = {}
        if prediction != 'all':
            filter_conditions.update({prediction: '> 0'})
        filter_conditions_list.append(filter_conditions)

    kwargs = {
        'image_id': image_id,
        'feature': args['feature'],
        'distance': args['distance'],
        'page': args['page'],
        'filter_conditions_list': filter_conditions_list,
        'results_per_page': 8
    }
    method_name = 'nn_by_id_many_filters'
    job = vislab.utils.redis_q.submit_job(
        method_name, kwargs, 'similarity_server')
    results_sets = vislab.utils.redis_q.get_return_value(job)

    for results_data, prediction in zip(results_sets, prediction_options):
        results_data['title'] = prediction

    # TODO: get this working again
    # image_info = collection.find_by_id(image_id, collection_name)
    image_info = df.loc[image_id].to_dict()

    return flask.render_template(
        'similarity.html', args=args,
        select_options=select_options,
        image_info=image_info,
        results_sets=results_sets
    )


if __name__ == '__main__':
    vislab.ui.util.start_from_terminal(app)
