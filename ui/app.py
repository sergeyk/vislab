"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- Display the number of results on the page.
- Switch to getting data from a mongo database instead of loading df.
"""
import flask
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
import vislab.datasets

app = flask.Flask(__name__)

pins_df = vislab.datasets.pinterest.get_pins_80k_df()
flickr_df = vislab.datasets.flickr.get_df()
style_names = vislab.datasets.flickr.underscored_style_names


@app.route('/')
def index():
     return flask.redirect(flask.url_for(
        'data', dataset_name='pinterest', style_name='style_Pastel',
        pins_per_user=5, page=1)
    )


@app.route('/data/<dataset_name>/<style_name>/<int:pins_per_user>/<int:page>')
def data(dataset_name, style_name, pins_per_user, page):
    if dataset_name == 'flickr':
        df = flickr_df
    elif dataset_name == 'pinterest':
        df = pins_df
    else:
        raise Exception("Unknown dataset")

    results_per_page = 7 * 20

    # Filter on style.
    if style_name != 'all':
        df = df[df[style_name]]

    # Filter on pins per user
    # TODO: bring this back
    # df = df.groupby('username').head(pins_per_user)
    # df.set_index(df.index.get_level_values(1), inplace=True)

    # Paginate
    num_pages = df.shape[0] / results_per_page
    start = page * results_per_page
    df = df.iloc[start:min(df.shape[0], start + results_per_page)]

    # Set filter options
    select_options = [
        ('dataset', ['flickr', 'pinterest'], dataset_name),
        ('style', ['all'] + style_names, style_name),
        ('pins_per_user', [1, 5, 100], pins_per_user),
        ('page', range(1, num_pages), page),
    ]

    # Fetch images and render.
    images = []
    for ix, row in df.iterrows():
        image_info = row.to_dict()
        images.append(image_info)

    return flask.render_template(
        'data.html', images=images, select_options=select_options
    )


if __name__ == '__main__':
    import sys
    debug = len(sys.argv) > 1 and sys.argv[1] == 'debug'
    if debug:
        print("Debug mode")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        http_server = HTTPServer(WSGIContainer(app))
        http_server.listen(5000)
        IOLoop.instance().start()
