"""
This file, vislab/ui/app.py, is the updated data and results viewing ui.
The older code for the same tasks is in vislab/app.py
We start with the Pinterest data here.

TODO
- switch from using dataframes for results to mongo
- make image page that displays all the true labels and pred values for
    a given image-
- rename 'index' to 'image_id'
"""
import flask
import tornado.wsgi
import tornado.httpserver
import numpy as np
import time
import pandas as pd
import vislab.datasets
import vislab._results

mongo_client = vislab.util.get_mongodb_client()
app = flask.Flask(__name__)
style_names = vislab.datasets.flickr.underscored_style_names
db_name = 'all_preds'
experiment_name = 'flickr_mar23'


def insert_df(df, collection):
    t = time.time()
    dict_list = []
    for i in range(df.shape[0]):
        if time.time() - t > 2.5:
            t = time.time()
            print('... on {}/{}'.format(i, df.shape[0]))

        d = df.iloc[i].to_dict()
        for k, v in d.iteritems():
            if type(d[k]) is np.bool_:
                d[k] = bool(d[k])
        dict_list.append(d)
        if i % 1000 == 0:
            print('inserting ....')
            collection.insert(dict_list)
            dict_list = []


def load_pred_results(results_dirname, experiment_name, settings):
    """
    Load prediction results into accessible form.
    Also, make positive predictions positive and negative negative, to
    simplify future sorting by confidence.
    """
    collection = mongo_client[db_name][experiment_name]

    if collection.count() == 0:
        _, preds_panel = vislab._results.load_pred_results(
            experiment_name, results_dirname,
            multiclass=True, force=False)
        settings += preds_panel.minor_axis.tolist()

        threshold_df = pd.read_hdf(
            '{}/{}_thresholds.h5'.format(results_dirname, experiment_name), 'df')

        for setting in settings:
            df_ = preds_panel.minor_xs(setting)
            for style_name in style_names:
                df_['pred_' + style_name] -= threshold_df.loc[style_name, setting]
                df_['abs_pred_' + style_name] = np.abs(df_['pred_' + style_name])
            df_['setting'] = setting
            df_ = df_.reset_index()

            insert_df(df_, collection)

    else:
        settings += collection.distinct('setting')

settings = []
results_dirname = vislab.config['paths']['shared_data'] + '/results_mar23'
load_pred_results(results_dirname, experiment_name, settings)
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
        insert_df(df, collection)
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

@app.route('/image/<int:img_id>')
def image_page(img_id):
    collection = mongo_client[db_name][experiment_name]
    page_url = flickr_df['page_url'][str(img_id)]
    image_url = flickr_df['image_url'][str(img_id)]
    fields = {'_id':0}
    #from IPython import embed
    #embed()
    for style in style_names:
        fields['pred_{}'.format(style)] = 1
    doc = collection.find({'index': str(img_id)}, fields)[0]
    df = pd.DataFrame({0: doc}).sort_index(by=[0], ascending=[False])
    table = df.to_html()
    table = str(table).replace("<th></th>\n",
            "<th>Style Prediction</th>\n", 1)
    table = table.replace("<th>0</th>\n",
            "<th>Confidence</th>\n", 1)
    conf = list(df[0])
    colors = [0] * len(conf)
    green = [0, 0, 0]
    pink = [255, 0, 0]
    for i in range(0, len(conf)):
        st = (abs(conf[i])/1.8)
        if conf[i] > 0:
            green[0] = (1-st)*180 + 18
            green[1] = (1-st)*80 + 118
            green[2] = (1-st)*180 + 18
            colors[i] = hex(green[0])[2:] + hex(green[1])[2:] + hex(green[2])[2:]
        else:
            pink[1] = 204 - st*70
            pink[2] = 204 - st*70
            colors[i] = hex(pink[0])[2:] + hex(pink[1])[2:] + hex(pink[2])[2:]
        table = table.replace("<th>pred_style_",
            "<th bgcolor='{}'>pred_style_".format(colors[i]), 1)
    return flask.render_template('image_page.html',
        image_url=image_url,
        page_url=page_url,
        table=table
    )



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
    t = time.time()
    collection = mongo_client[db_name][experiment]

    query = {'setting': str(setting)}
    sort_key = 'abs_pred_{}'.format(style)
    sort_dir = 1
    if split != 'all':
        query['split'] = str(split)

    if gt_label == 'positive':
        query[style] = True
    elif gt_label == 'negative':
        query[style] = False

    if pred_label == 'positive':
        query['pred_{}'.format(style)] = {"$gt": 0}
    elif pred_label == 'negative':
        query['pred_{}'.format(style)] = {"$lte": 0}

    if confidence == 'increasing':
        sort_dir = 1
    else:
        sort_dir = -1

    cursor = collection.find(query)
    cursor = cursor.sort(sort_key, sort_dir)
    num_results = cursor.count()
    results_per_page = 7 * 20
    num_pages = num_results / results_per_page

    start_ind = (page - 1) * results_per_page
    end_ind = min(num_results, start_ind + results_per_page)

    if num_results > 0:
        results = list(cursor[start_ind:end_ind])
        for result in results:
            #result['page_url'] = flickr_df['page_url'][result['index']]
            result['image_url'] = flickr_df['image_url'][result['index']]
            result['caption'] = 'conf: {:.2f} | gt: {}'.format(
                result['pred_' + style],
                '+' if result[style] else '-')
    else:
        results = []

    # Set filter options. Order matters.
    select_options = [
        ('experiment', ['flickr_mar23'], experiment),
        ('setting', settings, setting),
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
        page_type='results',
        time_elapsed=(time.time() - t)
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
