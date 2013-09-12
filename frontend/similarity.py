import flask
import vislab.frontend.util as util
import vislab.backend.collection
import aphrodite.flickr

app = flask.Flask(__name__)
collection = vislab.backend.collection.Collection()


@app.route('/')
def index():
    return flask.redirect(flask.url_for('similar_to'))


@app.route('/similar_to')
def similar_to_id():
    """
    This function does double duty: it returns both the rendered HTML
    and the JSON results, depending on whether the json arg is set.
    This keeps the parameter-parsing logic in one place.
    """
    select_options = {
        'distance': ['euclidean', 'manhattan', 'chi_square'],
        'style': ['all'] + aphrodite.flickr.underscored_style_names,
        'prediction': ['all'] + ['pred_{}'.format(x) for x in aphrodite.flickr.underscored_style_names]
    }

    args = util.get_query_args(
        necessary=['id'],
        defaults={
            'feature': 'decaf_fc6',
            'distance': 'euclidean',
            'page': 1,
            'style': 'all',
            'prediction': 'all'
        },
        types={
            'page': int
        }
    )

    # In case of style_pred args, make a lambda filter and pass it.
    # Want to keep collection code very domain-independent.
    filter_conditions = {}
    if args['style'] != 'all':
        filter_conditions = {args['style']: ''}
    if args['prediction'] != 'all':
        filter_conditions.update({args['prediction']: '> 0'})
    results_data = collection.nn_by_id(
        args['id'], args['feature'], args['distance'], args['page'],
        filter_conditions
    )

    # Fetch all information we have about the image: url, labels.
    image_info = collection.find_by_id(args['id'])

    return flask.render_template(
        'similarity.html', args=args,
        select_options=select_options,
        image_info=image_info,
        results_data=results_data
    )


if __name__ == '__main__':
    util.start_from_terminal(app)
