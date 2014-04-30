"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- switch from using dataframes for results to mongo
- make image page that displays all the true labels and pred values for
    a given image
"""
import flask
import tornado.wsgi
import tornado.httpserver
import numpy as np
import time
import pandas as pd
from collections import defaultdict
import vislab.datasets
import vislab._results

mongo_client = vislab.util.get_mongodb_client()
app = flask.Flask(__name__)
style_names = vislab.datasets.flickr.underscored_style_names


def load_pred_results(all_preds, results_dirname, experiment_name):
    """
    Load prediction results into accessible form.
    Also, make positive predictions positive and negative negative, to
    simplify future sorting by confidence.
    """
    threshold_df = pd.read_hdf(
        '{}/{}_thresholds.h5'.format(results_dirname, experiment_name), 'df')

    _, preds_panel = vislab._results.load_pred_results(
        experiment_name, results_dirname,
        multiclass=True, force=False)

    # TODO: load into mongo collections instead
    for setting in preds_panel.minor_axis:
        df_ = preds_panel.minor_xs(setting)
        for style_name in style_names:
            df_['pred_' + style_name] -= threshold_df.loc[style_name, setting]
            df_['abs_pred_' + style_name] = np.abs(df_['pred_' + style_name])
        all_preds[experiment_name][setting] = df_


results_dirname = vislab.config['paths']['shared_data'] + '/results_mar23'
all_preds = defaultdict(dict)  # TODO: use a mongo database instead
load_pred_results(all_preds, results_dirname, 'flickr_mar23')
flickr_df = vislab.datasets.flickr.get_df()  # TODO: get rid of this


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
                if type(d[k]) is np.bool_:
                    d[k] = bool(d[k])
            collection.insert(d)

    return collection


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/data')
def data_default():
    return flask.redirect(flask.url_for(
        'data', dataset_name='flickr', style_name='style_Pastel', page=1
    ))


@app.route('/results')
def results_default():
    return flask.redirect(flask.url_for(
        'results', experiment='flickr_mar23', setting='caffe_fc6 None vw',
        style='style_Vintage', split='test',
        gt_label='all', pred_label='positive', confidence='decreasing', page=1
    ))


@app.route('/results/<experiment>/<setting>/<style>/<split>/<gt_label>/<pred_label>/<confidence>/<int:page>')
def results(experiment, setting, style, split, gt_label, pred_label,
            confidence, page):
    """
    experiment:
        name of collection that held the results
        e.g. 'flickr_mar23'
    setting:
        features and special techniques used in classification
        e.g. 'caffe_fc6 None vw'
    style:
        name of predicted attribute
        e.g. 'style_Vintage'
    split:
        ['all', 'train', 'val', 'test']
    gt_label:
        ground truth value of style_name
        ['all', 'positive', 'negative']
    pred_label:
        predicted value of style_name
        ['all', 'positive', 'negative']
    confidence:
        ['increasing', 'decreasing']
    page:
        index of paginated results
    """
    # TODO: use mongo throughout

    df = all_preds[experiment][setting]

    if split != 'all':
        df = df[df['split'] == split]

    if gt_label == 'positive':
        df = df[df[style]]
    elif gt_label == 'negative':
        df = df[df[style] == False]

    if pred_label == 'positive':
        df = df[df['pred_' + style] > 0]
    elif pred_label == 'negative':
        df = df[df['pred_' + style] <= 0]

    if confidence == 'increasing':
        df.sort('abs_pred_' + style, ascending=True, inplace=True)
    else:
        df.sort('abs_pred_' + style, ascending=False, inplace=True)

    num_results = df.shape[0]
    results_per_page = 7 * 20
    num_pages = num_results / results_per_page

    start_ind = (page - 1) * results_per_page
    end_ind = min(num_results, start_ind + results_per_page)

    if num_results > 0:
        df = df.iloc[start_ind:end_ind]
        df['page_url'] = flickr_df['page_url']
        df['image_url'] = flickr_df['image_url']

        caption = lambda row: 'conf: {:.2f} | gt: {}'.format(
            row['pred_' + style],
            '+' if row[style] else '-'
        )
        df['caption'] = df.apply(caption, axis=1)
        results = [_.to_dict() for i, _ in df.iterrows()]
    else:
        results = []

    # Set filter options. Order matters.
    select_options = [
        ('experiment', ['flickr_mar23'], experiment),
        ('setting', all_preds[experiment].keys(), setting),
        ('style', style_names, style),
        ('split', ['all', 'train', 'val', 'test'], split),
        ('actual_label', ['all', 'positive', 'negative'], gt_label),
        ('predicted_label', ['all', 'positive', 'negative'], pred_label),
        ('confidence', ['increasing', 'decreasing'], confidence),
        ('page', range(1, num_pages), page)
    ]

    return flask.render_template(
        'data.html', images=results, select_options=select_options,
        num_results=num_results,
        start_results=results_per_page * (page - 1),
        end_results=results_per_page * page,
        page_type='results'
    )


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

    return flask.render_template(
        'data.html', images=results, select_options=select_options,
        num_results=num_results,
        start_results=results_per_page * (page - 1),
        end_results=results_per_page * page,
        page_type='data'
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
