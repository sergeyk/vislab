import flask
import vislab.frontend.util as util
import vislab.backend.collection
import aphrodite.flickr

app = flask.Flask(__name__)
collection = vislab.backend.collection.Collection()


@app.route('/')
def index():
    return flask.redirect(flask.url_for('similar_to_random'))


@app.route('/similar_to_random')
def similar_to_random():
    image_id = collection.get_random_id()
    return flask.redirect(flask.url_for(
        'similar_to_id', image_id=image_id))


@app.route('/similar_to/<image_id>')
def similar_to_id(image_id):
    """
    This function does double duty: it returns both the rendered HTML
    and the JSON results, depending on whether the json arg is set.
    This keeps the parameter-parsing logic in one place.
    """
    select_options = {
        'feature': ['deep learned fc6', 'style scores'],
        'distance': ['euclidean', 'manhattan'],
        'label': ['all'] + aphrodite.flickr.underscored_style_names,
        'prediction': ['all'] + ['pred_{}'.format(x) for x in aphrodite.flickr.underscored_style_names]
    }

    args = util.get_query_args(
        defaults={
            'feature': 'deep learned fc6',
            'distance': 'euclidean',
            'page': 1,
            'label': 'all',
            'prediction': 'all'
        },
        types={
            'page': int
        }
    )

    filter_conditions = {}
    if args['label'] != 'all':
        filter_conditions = {args['label']: ''}
    if args['prediction'] != 'all':
        filter_conditions.update({args['prediction']: '> 0'})
    results_data = collection.nn_by_id(
        image_id, args['feature'], args['distance'], args['page'],
        filter_conditions
    )

    image_info = collection.find_by_id(image_id)

    return flask.render_template(
        'similarity.html', args=args,
        select_options=select_options,
        image_info=image_info,
        results_data=results_data
    )


if __name__ == '__main__':
    util.start_from_terminal(app)
