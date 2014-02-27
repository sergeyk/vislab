import os
import pandas as pd
import cPickle
import numpy as np
import vislab


def load_pred_results(collection_name, cache_dirname, multiclass=False, force=False):
    """
    Return DataFrame of prediction experiment results and Panel of per-image
    predictions.
    """
    if not os.path.exists(cache_dirname):
        vislab.util.makedirs(cache_dirname)

    results_df_filename = os.path.join(
        cache_dirname, '{}_results_df.pickle'.format(collection_name))
    preds_panel_filename = os.path.join(
        cache_dirname, '{}_preds_panel.pickle'.format(collection_name))

    # If cache exists, load and return.
    if (os.path.exists(results_df_filename) and
            os.path.exists(preds_panel_filename) and
            not force):
        results_df = pd.read_pickle(results_df_filename)
        preds_panel = pd.read_pickle(preds_panel_filename)
        print("Loaded from cache: {} records".format(results_df.shape[0]))
        return results_df, preds_panel

    # Otherwise, construct from database.
    client = vislab.util.get_mongodb_client()
    collection = client['predict'][collection_name]
    print("Results in collection {}: {}".format(collection_name, collection.count()))

    df = pd.DataFrame(list(collection.find()))
    df.index = df.index.astype(str)

    # Make the features list hashable for filtering/joins.
    df['features_str'] = df['features'].apply(lambda x: ','.join(sorted(x)))

    # We need a unique representation of the predictor settings.
    df['setting'] = df.apply(lambda x: '{} {} {}'.format(x['features_str'], x['quadratic'], 'vw'), axis=1)

    # And of the task performed.
    df['full_task'] = df.apply(lambda x: '{} {}'.format(x['task'], x['data']), axis=1)

    df = df.drop_duplicates(cols=['features_str', 'full_task'], take_last=True)

    # Just for printing, if needed.
    df = df.sort(['full_task', 'setting'])

    # Get all predictions in a separate panel and drop the pickled ones.
    if multiclass:
        data = {}
        for setting in df['setting'].unique():
            el = df[df['setting'] == setting].iloc[0]
            try:
                pred_df = cPickle.loads(el['pred_df'])
            except:
                assert('results_name' in el)
                pred_df_filename = '{}/{}.h5'.format(
                    vislab.config['paths']['results'], el['results_name'])
                #print(pred_df_filename)
                pred_df = pd.read_hdf(pred_df_filename, 'df')

            # Not sure why there should ever be duplicate indices, but
            # there are for one of the wikipaintings results...
            pred_df['__index'] = pred_df.index
            pred_df.drop_duplicates(cols='__index', take_last=True, inplace=True)
            del pred_df['__index']

            data[setting] = pred_df

        preds_panel = pd.Panel(data).swapaxes('items', 'minor')

    else:
        preds_panel = get_all_preds_panel(df)

    try:
        del df['pred_df']
    except KeyError:
        pass

    df.to_pickle(results_df_filename)
    preds_panel.to_pickle(preds_panel_filename)

    return df, preds_panel


def get_all_preds_panel(df):
    all_full_tasks = df['full_task'].unique()
    data = dict((
        (full_task, get_all_preds_df(df, full_task))
        for full_task in all_full_tasks
    ))
    all_preds_panel = pd.Panel(data)
    return all_preds_panel


def get_all_preds_df(df, full_task):
    """
    Get the DataFrame of predictions from the results dataframe.

    Tip: get all predictions of an image with
        all_preds_panel.major_xs('f_1604904579').T
    """
    tdf = df[df['full_task'] == full_task]

    # Make sure that there are no duplicate settings.
    if len(tdf.setting.unique()) != tdf.shape[0]:
        try:
            del df['pred_df']
        except KeyError:
            pass
        print(tdf.to_string())
        raise Exception("Non-unique feature-setting pairs")

    pred_dfs = []
    for i, row in tdf.iterrows():
        try:
            pred_df = cPickle.loads(row['pred_df'])
        except:
            assert('results_name' in row)
            pred_df_filename = '{}/{}.h5'.format(
                vislab.config['paths']['results'], row['results_name'])
            pred_df = pd.read_hdf(pred_df_filename, 'df')
        pred_df.index = pred_df.index.astype(str)
        pred_dfs.append(pred_df)

    # Make sure that all the settings had the same label and split information
    arbitrary_pred_df = pred_dfs[0]
    assert(np.all(df_['label'] == arbitrary_pred_df['label'] for df_ in pred_dfs))
    assert(np.all(df_['split'] == arbitrary_pred_df['split'] for df_ in pred_dfs))

    data = []
    for df_ in pred_dfs:
        df_["index"] = df_.index
        # TODO: why the fuck are the duplicate indices???
        df_ = df_.drop_duplicates('index')
        if 'score' in df_.columns:
            data.append(df_['score'])
        else:
            # TODO: temporary, remove when all experiments are re-run
            data.append(df_['pred'])
    all_preds_df = pd.DataFrame(data, index=tdf['setting']).T

    all_preds_df['label'] = arbitrary_pred_df['label']
    all_preds_df['split'] = arbitrary_pred_df['split']

#     # Re-order columns
# #    columns = all_preds_df.columns.values
# #    reordered_columns = ['split', 'label'] + (columns - ['split', 'label']).tolist()
# #    all_preds_df = all_preds_df[:, reordered_columns]

    all_preds_df.index = all_preds_df.index.astype(str)

    return all_preds_df


if __name__ == '__main__':
    load_pred_results('wikipaintings_oct25', 'whatever', multiclass=True)
