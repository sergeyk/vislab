"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- port the result-displaying capability from old vislab/app.py
"""
import flask
import tornado.wsgi
import tornado.httpserver
import numpy
import time
import vislab.datasets

mongo_client = vislab.util.get_mongodb_client()
app = flask.Flask(__name__)
style_names = vislab.datasets.flickr.underscored_style_names


@app.route('/')
def index():
    """
    Redirect to the data page for Flickr:Pastel.
    """
    return flask.redirect(flask.url_for(
        'data', dataset_name='flickr', style_name='style_Pastel', page=1)
    )


def get_collection(dataset_name):
    """
    Return MongoDB collection for the given dataset_name.
    If not already present, then load the DataFrame from cache and
    insert into Mongo.
    """
    dataset_loaders = {
        'flickr': vislab.datasets.flickr.get_df,
        'pinterest': vislab.datasets.pinterest.get_pins_80k_df
    }

    collection = mongo_client['ui_datasets'][dataset_name]
    if collection.count() == 0:
        print "Inserting dataset DF for {}".format(dataset_name)
        df = dataset_loaders[dataset_name]()

        t = time.time()
        for i in range(df.shape[0]):
            if time.time() - t > 2.5:
                t = time.time()
                print('... on {}/{}'.format(i, df.shape[0]))

            d = df.iloc[i].to_dict()
            for k, v in d.iteritems():
                if type(d[k]) is numpy.bool_:
                    d[k] = bool(d[k])
            collection.insert(d)

    return collection


@app.route('/data/<dataset_name>/<style_name>/<int:page>')
def data(dataset_name, style_name, page):
    db = get_collection(dataset_name)

    results_per_page = 7 * 20

    # Filter on style.
    if style_name != 'all':
        cursor = db.find({style_name: True})
    else:
        cursor = db.find()

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
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app))
        http_server.listen(5000)
        tornado.ioloop.IOLoop.instance().start()
