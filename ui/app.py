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
import numpy
import time

client = vislab.util.get_mongodb_client()
app = flask.Flask(__name__)
style_names = vislab.datasets.flickr.underscored_style_names


@app.route('/')
def index():
    return flask.redirect(flask.url_for(
        'data', dataset_name='pinterest', style_name='style_Pastel',
        pins_per_user=5, page=1)
    )


def get_database(dataset_name):
    """
    Return MongoDB database for the given dataset_name
    Load the DataFrame and insert into Mongo, if not already present.
    """
    dataset_info = {
        'flickr': vislab.datasets.flickr.get_df,
        'pinterest': vislab.datasets.pinterest.get_pins_80k_df
    }

    db = client['ui_dfs'][dataset_name]
    if db.count() == 0:
        print "Inserting DF for {}".format(dataset_name)
        try:
            df = dataset_info[dataset_name]()
        except:
            raise Exception("Unknown dataset: {}".format(dataset_name))
        t = time.time()
        for i in range(df.shape[0]):
            if time.time() - t > 2.5:
                print('... on {}/{}'.format(i, df.shape[0]))
                t = time.time()
            d = df.iloc[i].to_dict()
            for k, v in d.iteritems():
                if type(d[k]) is numpy.bool_:
                    d[k] = bool(d[k])
            db.insert(d)
    return db


@app.route('/data/<dataset_name>/<style_name>/<int:pins_per_user>/<int:page>')
def data(dataset_name, style_name, pins_per_user, page):
    db = get_database(dataset_name)

    results_per_page = 7 * 20

    # Filter on style.
    if style_name != 'all':
        cursor = db.find({style_name: True})
    else:
        cursor = db.find()

    # Filter on pins per user
    # TODO: bring this back
    # df = df.groupby('username').head(pins_per_user)
    # df.set_index(df.index.get_level_values(1), inplace=True)

    # Paginate
    num_results = cursor.count()
    num_pages = num_results / results_per_page
    start = page * results_per_page
    end = min(num_results, start + results_per_page)
    results = cursor[start:end]

    # Set filter options
    select_options = [
        ('dataset', ['flickr', 'pinterest'], dataset_name),
        ('style', ['all'] + style_names, style_name),
        ('pins_per_user', [1, 5, 100], pins_per_user),
        ('page', range(1, num_pages), page),
    ]

    # Fetch images and render.
    images = results

    return flask.render_template(
        'data.html', images=images, select_options=select_options,
        num_results=num_results,
        start_results=results_per_page * (page - 1),
        end_results=results_per_page * page
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
