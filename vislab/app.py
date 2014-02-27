import os
import flask
import json
import pandas as pd
import vislab
import vislab.datasets


def make_json_response(body, status_code=200):
    resp = flask.make_response(json.dumps(body))
    resp.status_code = status_code
    resp.mimetype = 'application/json'
    return resp


app = flask.Flask(__name__)

## Load all kinds of stuff
df = vislab.datasets.ava.get_ava_df()
url_df = vislab.datasets.ava.get_urls_df()
style_df = vislab.datasets.ava.get_style_df()
flickr_df = vislab.datasets.flickr.load_flickr_df()
wp_df = vislab.datasets.wikipaintings.get_style_df()
wp_df = wp_df.join(vislab.datasets.wikipaintings.get_df())
behance_df = vislab.datasets.behance.get_photo_df()
behance_df['image_url'] = behance_df['imageURL']
behance_df['page_url'] = '#'
behance_illustration_df = vislab.datasets.behance.get_illustration_df()
behance_illustration_df['page_url'] = behance_illustration_df.apply(
    lambda x: 'http://www.behance.net/gallery/null/{}'.format(x['project_id']),
    axis=1
)
# behance_illustration_df = behance_illustration_df[
#     [not(x.endswith('.gif') or x.endswith('.png')) for x in behance_illustration_df['image_url']]
# ]

## Load results and predictions.
results_dir = 'data/results'
collection_names = {
    'flickr': 'aug29',
    'ava': 'final2',
    'wikipaintings': 'wikipaintings_sep26',
    'behance': 'behance',
    'behance_illustration': 'behance_dec28'
}

preds_panels = {}
setting_options = {}
task_options = {}
for dataset, collection_name in collection_names.iteritems():
    # Write out the results so that we can serve in a format that d3 can parse.
    results_df_filename = os.path.join(
        results_dir, '{}_results_df.pickle'.format(collection_name))
    results_df = pd.read_pickle(results_df_filename)
    results_table = results_df.pivot(
        index='full_task', columns='setting', values='score_test')
    results_table_filename = 'vislab/static/{}_results_table.csv'.format(dataset)
    results_table.to_csv(results_table_filename)

    setting_options[dataset] = results_df['setting'].unique().tolist()
    task_options[dataset] = results_df['full_task'].unique().tolist()

    # Load the prediction table into memory.
    preds_panel_filename = os.path.join(
        results_dir, '{}_preds_panel.pickle'.format(collection_name))
    preds_panels[dataset] = pd.read_pickle(preds_panel_filename)

    if dataset == 'behance':
        p = preds_panels[dataset]
        # TODO: get this into a DataFrame of same format as other datasets
        styles = [x for x in p.items if x.startswith('style_')]
        dfs = {}
        for style in styles:
            df = pd.DataFrame(index=p[style].index)
            df['label'] = -1
            df['label'][p[style]['decaf_fc6 False vw']] = 1
            df['decaf_fc6 False vw'] = p['pred_' + style]['decaf_fc6 False vw']
            df['split'] = 'test'
            dfs[style] = df
        preds_panels[dataset] = pd.Panel(dfs)
        task_options[dataset] = styles


@app.route('/<dataset_name>/results_table')
def results_table(dataset_name):
    return flask.send_file('static/{}_results_table.csv'.format(dataset_name))


@app.route('/<dataset_name>/images_json')
def images_json(dataset_name):
    args = get_query_args(data_query_arg_defaults[dataset_name])
    results = get_images('data', dataset_name, **args)
    return make_json_response(results)


@app.route('/<dataset_name>/results_images_json')
def results_images_json(dataset_name):
    args = get_query_args(get_default_query_args_for_results(dataset_name))
    results = get_images('results', dataset_name, **args)
    return make_json_response(results)


def get_images(mode, dataset_name, **args):
    assert(mode in ['data', 'results'])

    if mode == 'results':
        # Get a DataFrame of predictions for the given task.
        preds_panel = preds_panels[dataset_name]
        sdf = preds_panel[args['task']]
    else:
        if dataset_name == 'ava':
            sdf = df
        else:
            sdf = flickr_df

    if mode == 'data' and dataset_name == 'flickr':
        sdf = flickr_df

        if args['style'] not in [None, 'None', 'all']:
            sdf = sdf[sdf[args['style']]]

    if mode == 'data' and dataset_name == 'wikipaintings':
        sdf = wp_df

        if args['style'] not in [None, 'None', 'all']:
            sdf = sdf[sdf[args['style']]]

        if args['genre'] not in [None, 'None', 'all']:
            sdf = sdf[sdf[args['genre']]]

    if mode == 'data' and dataset_name == 'ava':
        # Filter by rating, if given.
        if args['rating_mean_min'] not in [None, '']:
            sdf = sdf[sdf['rating_mean'] >= args['rating_mean_min']]
        if args['rating_mean_max'] not in [None, '']:
            sdf = sdf[sdf['rating_mean'] <= args['rating_mean_max']]
        if args['rating_std_min'] not in [None, '']:
            sdf = sdf[sdf['rating_std'] >= args['rating_std_min']]
        if args['rating_std_max'] not in [None, '']:
            sdf = sdf[sdf['rating_std'] <= args['rating_std_max']]

        # Filter on tag, if given.
        if args['tag'] not in [None, 'None', 'all']:
            sdf = sdf[(sdf['semantic_tag_1_name'] == args['tag']) |
                      (sdf['semantic_tag_2_name'] == args['tag'])]

        # Filter on style, if given.
        if args['style'] not in [None, 'None', 'all']:
            sdf = sdf.ix[style_df[style_df[args['style']]].index]

        # Sort.
        if args['sort'] not in [None, 'None']:
            if args['sort'] == 'Highest average rating (best) first':
                sdf = sdf.sort('rating_mean', ascending=False)
            elif args['sort'] == 'Lowest average rating (worst) first':
                sdf = sdf.sort('rating_mean', ascending=True)
            elif args['sort'] == 'Highest rating variance (most contentious) first':
                sdf = sdf.sort('rating_std', ascending=False)
            elif args['sort'] == 'Lowest rating variance (least contentious) first':
                sdf = sdf.sort('rating_std', ascending=True)
            else:
                raise Exception("Unknown sort mode.")

    if mode == 'results':
        # Filter the predictions by argeter setting, if given.
        if args['setting'] not in [None, 'None']:
            sdf['selected_pred'] = sdf[args['setting']]

            num = sdf.shape[0]
            sdf = sdf.dropna(subset=['selected_pred'])
            print("Dropped {} results due to missing data".format(
                num - sdf.shape[0]))

        # Filter by data split, if given.
        if 'split' not in sdf.columns:
            sdf['split'] = 'test'

        if args['split'] not in [None, 'None']:
            sdf = sdf[sdf['split'] == args['split']]

        # Filter by data point label, if not set to 'all'.
        if 'label' not in sdf.columns:
            sdf['label'] = 'negative'

        if args['label'] == 'positive':
            sdf = sdf[sdf['label'] == 1]
        elif args['label'] == 'negative':
            sdf = sdf[sdf['label'] == -1]

        # Add a new column of binarized scores.
        sdf['selected_pred_binarized'] = -1
        sdf['selected_pred_binarized'][sdf['selected_pred'] > 0] = 1

        # Filter by prediction, if not set to 'all'.
        if args['prediction'] == 'positive':
            sdf = sdf[sdf['selected_pred_binarized'] == 1]
        elif args['prediction'] == 'negative':
            sdf = sdf[sdf['selected_pred_binarized'] == -1]

        # Sort by prediction confidence.
        if args['result_sort'] == 'increasing confidence':
            sdf = sdf.iloc[sdf['selected_pred'].abs().argsort().values]
        else:
            sdf = sdf.iloc[sdf['selected_pred'].abs().argsort()[-1::-1].values]

    # Client will deal with the 500.
    num_results = sdf.shape[0]
    if num_results == 0:
        return flask.abort(500)

    per_page = 8 * 12  # images are displayed in rows of 8
    start_ind = (args['page'] - 1) * per_page
    end_ind = min(num_results, start_ind + per_page)
    sdf = sdf.iloc[start_ind:end_ind]

    # Join to a table of metadata, such as image_url.
    # TODO: make this more elegant, or simply pre-join on load.
    if dataset_name == 'ava':
        sdf = sdf.join(url_df)

        if mode == 'results':
            sdf = sdf.join(df)

    if mode == 'results' and dataset_name == 'flickr':
        sdf = sdf.join(flickr_df)

    if mode == 'results' and dataset_name == 'wikipaintings':
        sdf = sdf.join(wp_df)

    if mode == 'results' and dataset_name == 'behance':
        sdf = sdf.join(behance_df, rsuffix='_')

    if mode == 'results' and dataset_name == 'behance_illustration':
        sdf = sdf.join(behance_illustration_df, rsuffix='_')

    # Assemble a list of results to return as JSON.
    results = []
    for image_id, row in sdf.iterrows():
        result = {
            'image_id': image_id,
            'image_url': row['image_url'],
            'page_url': row['page_url'],
        }

        if dataset_name == 'ava':
            result.update({
                'rating_mean': row['rating_mean'],
                'rating_std': row['rating_std'],
                'ratings': ','.join(str(i) for i in row['ratings'])
            })

        if mode == 'results':
            result.update({
                'split': row['split'],
                'label': row['label'],
                'selected_pred': row['selected_pred'],
                'selected_pred_binarized': row['selected_pred_binarized']
            })

        results.append(result)

    images_data = {
        'results': results,
        'start': start_ind,
        'num_results': num_results
    }
    return images_data


def get_default_query_args_for_results(dataset_name):
    defaults = {
        'page': 1,
        'setting': 'decaf_fc6  vw',
        'split': 'test',
        'label': 'all',
        'prediction': 'positive',
        'result_sort': 'decreasing confidence'
    }
    default_tasks = {
        'flickr': 'clf flickr_style_Bright,_Energetic',
        'ava': 'clf rating_mean',
        'wikipaintings': 'clf wikipaintings_style_style_Baroque',
        'behance': 'style_Vintage',
        'behance_illustration': 'clf behance_illustration_tag_3d'
    }
    defaults['task'] = default_tasks[dataset_name]
    if dataset_name == 'behance':
        defaults['setting'] = 'decaf_fc6 False vw'
    if dataset_name == 'behance_illustration':
        defaults['setting'] = 'decaf_fc6 None vw'
    return defaults


def get_query_args(defaults=None):
    args = dict((key, val) for key, val in flask.request.args.iteritems() if val not in ['None', ''])
    print("get_query_args: Request args: {}".format(args))

    # Update the dictionary with default values if they are missing.
    if defaults is not None:
        for key, val in defaults.items():
            if key not in args or args[key] is None:
                args[key] = val

    # Set data types correctly.
    for arg in ['page']:
        if arg in args and args[arg] is not None:
            args[arg] = int(args[arg])

    for arg in [
            'rating_mean_min', 'rating_mean_max',
            'rating_std_min', 'rating_std_max']:
        if arg in args and args[arg] not in [None, '']:
            args[arg] = float(args[arg])

    return args


@app.route('/')
def index():
    return flask.render_template('index.html')


data_query_arg_defaults = {
    'ava': {
        'page': 1,
        'style': 'all',
        'tag': 'all',
        'sort': 'Highest average rating (best) first',
        'rating_mean_min': None,
        'rating_mean_max': None,
        'rating_std_min': None,
        'rating_std_max': None
    },
    'flickr': {
        'page': 1,
        'style': 'all'
    },
    'wikipaintings': {
        'page': 1,
        'style': 'all',
        'genre': 'all'
    },
    'behance': {
        'page': 1,
        'style': 'all'
    },
    'behance_illustration': {
        'page': 1,
        'style': 'all'
    }
}


@app.route('/ava/data')
def ava_data():
    select_options = {
        'sort': [
            'Highest average rating (best) first',
            'Lowest average rating (worst) first',
            'Highest rating variance (most contentious) first',
            'Lowest rating variance (least contentious) first'
        ],
        'style': ['all'] + style_df.columns.tolist(),
        'tag': ['all'] + df.semantic_tag_1_name.dropna().unique().tolist()
    }
    args = get_query_args(data_query_arg_defaults['ava'])
    return flask.render_template(
        'results.html', select_options=select_options, args=args,
        page_title='AVA data', page_mode='data', dataset_name='ava'
    )


@app.route('/flickr/data')
def flickr_data():
    select_options = {
        'style': ['all'] + vislab.datasets.flickr.underscored_style_names
    }
    args = get_query_args(data_query_arg_defaults['flickr'])
    return flask.render_template(
        'results.html', select_options=select_options, args=args,
        page_title='Flickr data', page_mode='data', dataset_name='flickr'
    )


@app.route('/wikipaintings/data')
def wikipaintings_data():
    style_names = [x for x in wp_df.columns if x[:6] == 'style_']
    genre_names = [x for x in wp_df.columns if x[:6] == 'genre_']
    select_options = {
        'style': ['all'] + style_names,
        'genre': ['all'] + genre_names
    }
    args = get_query_args(data_query_arg_defaults['wikipaintings'])
    return flask.render_template(
        'results.html', select_options=select_options, args=args,
        page_title='WikiPaintings data', page_mode='data',
        dataset_name='wikipaintings'
    )


@app.route('/<dataset_name>/results')
def results(dataset_name):
    select_options = {
        'task': task_options[dataset_name],
        'setting': setting_options[dataset_name],
        'split': ['train', 'val', 'test'],
        'label': ['all', 'positive', 'negative'],
        'prediction': ['all', 'positive', 'negative'],
        'result_sort': ['decreasing confidence', 'increasing confidence'],
    }
    args = get_query_args(get_default_query_args_for_results(dataset_name))
    return flask.render_template(
        'results.html', args=args,
        select_options=select_options,
        page_title='{} results dashboard'.format(dataset_name),
        page_mode='results', dataset_name=dataset_name)


@app.route('/<dataset_name>/image')
def image(dataset_name):
    image_id = flask.request.args.get('image_id')

    image_info = {}
    if dataset_name == 'ava':
        image_info.update({
            'image_url': url_df['image_url'].ix[image_id],
            'page_url': vislab.image.AVA_URL_FOR_ID.format(image_id)
        })
    elif dataset_name == 'flickr':
        image_info.update({
            'image_url': flickr_df['image_url'].ix[image_id],
            'page_url': flickr_df['page_url'].ix[image_id],
        })
    elif dataset_name == 'wikipaintings':
        image_info.update({
            'image_url': wp_df['image_url'].ix[image_id],
            'page_url': wp_df['page_url'].ix[image_id],
        })
    elif dataset_name == 'behance':
        image_info.update({
            'image_url': behance_df['imageURL'].ix[image_id],
            'page_url': '#',
        })
    elif dataset_name == 'behance_illustration':
        image_info.update({
            'image_url': behance_illustration_df['image_url'].ix[image_id],
            'page_url': behance_illustration_df['page_url'].ix[image_id],
        })
    else:
        raise Exception('Uknown dataset')

    if dataset_name in preds_panels:
        preds_panel = preds_panels[dataset_name]
        if image_id in preds_panel.major_axis:
            preds = preds_panel.major_xs(image_id).T
            try:
                # TODO: sort by best-performing feature
                preds = preds.sort('decaf_fc6  vw', ascending=False)
            except:
                pass
            image_info['preds_table'] = flask.Markup(
                preds.to_html(float_format=lambda x: '{:.3f}'.format(x)))

    return flask.render_template(
        'images.html',
        image_info=image_info
    )


if __name__ == '__main__':
    # Local dev
    if os.path.exists('/Users/sergeyk'):
        app.run(debug=True, host='0.0.0.0', port=5000)

    # ICSI
    elif os.path.exists('/u/sergeyk'):
        app.run(debug=False, host='0.0.0.0', port=5000)

    # EC2 or something.
    else:
        app.run(debug=False, host='0.0.0.0', port=80)
