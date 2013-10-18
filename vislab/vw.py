"""
Train and test Vowpal Wabbit classifier or regressor on data.
Training includes cross-validation over parameters.
"""
import time
import os
import pandas
import socket
import cPickle
import bson
import sklearn.metrics
import sklearn.grid_search
import numpy as np
import vislab
import vislab.results

# The base VW command: run quiet, compress the cache, and use more than
# the default 18 bits.
# TODO: make bit_precision a commandline argument
vw_cmd = "vw --quiet --compressed --bit_precision=24"


def train_and_test(
        collection_name, dataset, feature_names,
        force=False, num_workers=4,
        num_passes=[10], loss=['logistic'], l1_weight=[0], l2_weight=[0],
        quadratic='', verbose=False):
    """
    Train and test using VW with the given features and quadratic
    expansion.
    Features are assumed to be stored in VW format in canonical location.
    Cross-validates over all combinations of given parameter choices.

    Parameters
    ----------
    collection_name: string
        Name of the MongoDB 'predict' database collection that contains
        the prediction results.
    dataset: dict
        Contains name information and DataFrames for train, val,
        and test splits.
    feature_names: list of string
        Features to use.
    force: boolean [False]
    num_workers: int [4]
        VW parameter tuning will run in parallel with this many workers,
        on the same machine and reading from the same cache file.
    num_passes: list of int
    loss: list of string [['logistic']]
        Acceptable choices are [
            'hinge', 'squared', 'logistic', 'quantile'
        ]
    l1_weight: list of float [[0]]
    l2_weight: list of float [[0]]
    quadratic: string ['']
        If a non-empty string is given, it must be a sequence of single
        letter corresponding to the namespaces that will be crossed, or
        the word 'all'.
    verbose: boolean [False]
    """
    print("{} running VW on {} for {}".format(
        socket.gethostname(), dataset['name'], dataset['task']))
    print("Using {}, quadratic: {}".format(feature_names, quadratic))

    # To check for existing record of the final score, we construct a
    # query document.
    document = {
        'features': feature_names,
        'task': dataset['task'],
        'quadratic': quadratic
    }
    # The salient parts include the type of data: 'rating', 'style', etc.
    document.update(dataset['salient_parts'])

    # The results are stored in a Mongo database called 'predict'.
    client = vislab.util.get_mongodb_client()
    collection = client['predict'][collection_name]
    if not force:
        result = collection.find_one(document)
        if result is not None:
            print("Already classified this, not doing anything.")
            print(document)
            print("(Score was {:.3f})".format(result['score_test']))
            return

    dirname = vislab.util.makedirs('{}/{}'.format(
        vislab.config['paths']['predict'], dataset['name']))
    splits = ['train', 'val', 'test']

    # Save the dataset DataFrames for use in filtering examples.
    t = time.time()
    df_filenames = {}
    for name in splits:
        df_filenames[name] = '{}/{}_df.h5'.format(dirname, name)
        dataset['{}_df'.format(name)].to_hdf(df_filenames[name], 'df')
    print('Saving DataFrames for splits took {:.3f} s'.format(time.time() - t))

    # Set canonical location for VW feature cache files.
    cache_filenames = [
        '{}/{}/{}.txt.gz'.format(
            vislab.config['paths']['feats'], dataset['dataset_name'], f)
        for f in feature_names
    ]

    # Cache train, val, and test data to own folders.
    t = time.time()
    out_dirnames = {}
    for name in splits:
        experiment_name = '_'.join(feature_names) + '_q_{}'.format(quadratic)
        out_dirnames[name] = vislab.util.makedirs(
            '{}/{}/{}'.format(dirname, experiment_name, name))
        _cache_data(
            df_filenames[name], cache_filenames, out_dirnames[name],
            verbose, force)
    print('Caching all data took {:.3f} s'.format(time.time() - t))

    # Train several models, return the one with best validation performance.
    param_grid = {
        'loss': loss,
        'num_passes': num_passes,
        'l1_weight': l1_weight,
        'l2_weight': l2_weight,
        'quadratic': [quadratic]
    }
    grid = list(sklearn.grid_search.ParameterGrid(param_grid))

    t = time.time()
    best_setting, best_val_score, train_pred_df, val_pred_df = _train_with_val(
        dataset, out_dirnames, cache_filenames, df_filenames, grid,
        num_workers, verbose)
    print('Training all models took {:.3f} s'.format(time.time() - t))

    # Train the best model with more data: the validation set.
    t = time.time()
    print("Updating best VW model with validation data")
    best_model_filename = '{}/{}_model.vw'.format(
        out_dirnames['train'], _setting_to_name(**best_setting))
    cmd = _train_vw_cmd(
        best_setting, out_dirnames['val'], from_model=best_model_filename)
    cmd_filename = out_dirnames['val'] + '/_train_cmd.sh'
    vislab.util.run_through_bash_script([cmd], cmd_filename, verbose)

    print("Running VW prediction")
    pred_cmd = _pred_vw_cmd(
        best_setting, out_dirnames['val'], out_dirnames['test'])
    pred_filename = '{}/{}_pred.txt'.format(
        out_dirnames['test'], _setting_to_name(**best_setting))
    cmd_filename = out_dirnames['test'] + '/_test_cmd.sh'
    vislab.util.run_through_bash_script([pred_cmd], cmd_filename, verbose)

    # Test the best model updated with val data on the test set.
    test_pred_df, score = _get_preds_and_score(
        pred_filename, dataset['test_df'],
        dataset['task'], dataset['num_labels'])

    # Combine all the predictions into one DataFrame
    train_pred_df['split'] = 'train'
    val_pred_df['split'] = 'val'
    test_pred_df['split'] = 'test'
    pred_df = train_pred_df.append(val_pred_df).append(test_pred_df)

    original_document = document.copy()
    document.update({
        'score_test': score,
        'score_val': best_val_score,
        'pred_df': bson.Binary(cPickle.dumps(pred_df, protocol=2))
    })
    collection.update(original_document, document, upsert=True)
    print("Final score: {:.3f}".format(score))
    print('Final prediction took {:.3f} s'.format(time.time() - t))


def _cache_data(
        labels_df_filename, feature_filenames, output_dirname,
        verbose=False, force=False):
    """
    Run the labels and feature data through VW once to output to cache.

    Parameters
    ----------
    labels_df_filename: string
        Contains a DataFrame whose index is image ids.
    feature_filenames: list of string
    output_dirname: string
    verbose: boolean [False]
    force: boolean [False]
    """
    cache_filename = output_dirname + '/cache.vw'
    cache_preview_filename = output_dirname + '/cache_preview.txt'
    if not force and os.path.exists(cache_filename):
        print("Cache file exists, not doing anything.")
        return

    # Concatenate the feature files horizontally.
    zcats = ' '.join('<(zcat {})'.format(f) for f in feature_filenames)
    paste_cmd = "paste -d'\\0' {}".format(zcats)

    # The output of the above is piped through a filter that selects by id.
    vw_filter_filename = vislab.repo_dirname + '/vw_filter.py'
    filter_cmd = "python {} {}".format(vw_filter_filename, labels_df_filename)

    # Output a few lines of what is piped into VW to a preview file.
    head_cmd = 'head -n 10'
    cache_preview_cmd = "{} | {} | {} > {}".format(
        paste_cmd, filter_cmd, head_cmd, cache_preview_filename)

    # Run all data through VW to cache it.
    cache_cmd = "{} | {} | {} -k --noop --cache_file {}".format(
        paste_cmd, filter_cmd, vw_cmd, cache_filename)

    print("Caching data")
    cmd_filename = output_dirname + '/_cache_cmd.sh'
    vislab.util.run_through_bash_script(
        [cache_preview_cmd, cache_cmd], cmd_filename, verbose)


def _setting_to_name(loss, num_passes, l1_weight, l2_weight, quadratic):
    name = '{}_np_{}_l1_{:.9f}_l2_{:.9f}_q_{}'.format(
        loss, num_passes, l1_weight, l2_weight, quadratic)
    return name


def _train_vw_cmd(setting, dirname, from_model=None):
    """
    Return command to train VW.

    Parameters
    ----------
    setting: dict
        Parameters for VW.
    dirname: string
        Directory containing cached data.
    from_model: string or None [None]
        If given, begins training from this model.
    """
    model_filename = '{}/{}_model.vw'.format(
        dirname, _setting_to_name(**setting))

    # The basic command.
    cmd = vw_cmd + " -f {} --cache_file={}/cache.vw".format(
        model_filename, dirname)
    cmd += " --l1={:.9f} --l2={:.9f} --passes={} --loss_function={}".format(
        setting['l1_weight'], setting['l2_weight'],
        setting['num_passes'], setting['loss'])

    # If we are training from scratch, then we will use all data available,
    # because we have a dedicated val set.
    if from_model is None:
        cmd += " --holdout_off --save_resume"

    # If we are training with the val data starting with the best model,
    # then we turn on early termination and set the holdout fraction to
    # 1/8, so that we don't overfit.
    else:
        cmd += " --early_terminate=3 --holdout_period=8 -i {}".format(
            from_model)

    # Append the quadratic setting.
    quadratic = setting['quadratic']
    if quadratic == 'all':
        quadratic = '::'
    if len(quadratic) > 0:
        cmd += " --quadratic={}".format(quadratic)

    return cmd


def _pred_vw_cmd(setting, model_dirname, pred_dirname):
    """
    Return command to predict with a VW model.

    Parameters
    ----------
    setting: dict
        VW settings.
    model_dirname: string
        Directory containing model file with the given settings.
    pred_dirname: string
        Directory containing cache files. Predictions will be output
        here as well.
    """
    name = _setting_to_name(**setting)
    model_filename = '{}/{}_model.vw'.format(model_dirname, name)
    pred_filename = '{}/{}_pred.txt'.format(pred_dirname, name)
    cmd = vw_cmd + " -t -i {} --cache_file={}/cache.vw -p {}".format(
        model_filename, pred_dirname, pred_filename)
    return cmd


def _train_with_val(
        dataset, out_dirnames, cache_filenames, df_filenames,
        grid, num_workers=4, verbose=False):
    """
    Run training using the cache.
    See docstring to train().

    Returns
    -------
    best_model_filename: string
    best_score: float
    """
    train_dir = out_dirnames['train']
    val_dir = out_dirnames['val']

    # Assemble training, training prediction, and val prediction cmds.
    train_cmds, train_pred_cmds, val_pred_cmds = zip(*[
        (
            _train_vw_cmd(setting, train_dir),
            _pred_vw_cmd(setting, train_dir, train_dir),
            _pred_vw_cmd(setting, train_dir, val_dir)
        )
        for setting in grid
    ])

    # The idea is that each VW uses at most two cores and very little memory,
    # and so can run in parallel with other instances. Since the instances
    # read data at roughly the same rate, they should benefit from OS caching.
    parallel_cmds = [
        "echo \"{}\" | parallel -j {}".format('\n'.join(cmds), num_workers)
        for cmds in [train_cmds, train_pred_cmds, val_pred_cmds]
    ]

    print("Running VW training for {} param settings, {} at a time".format(
        len(grid), num_workers))
    cmd_filename = out_dirnames['train'] + '/_train_cmd.sh'
    vislab.util.run_through_bash_script(
        parallel_cmds, cmd_filename, verbose=False)

    # Load all the model predictions and pick the best settings.
    train_scores = []
    val_scores = []
    train_pred_dfs = []
    val_pred_dfs = []
    for setting in grid:
        train_pred_df, train_score = _get_preds_and_score(
            '{}/{}_pred.txt'.format(
                out_dirnames['train'], _setting_to_name(**setting)),
            dataset['train_df'], dataset['task'], dataset['num_labels'])

        val_pred_df, val_score = _get_preds_and_score(
            '{}/{}_pred.txt'.format(
                out_dirnames['val'], _setting_to_name(**setting)),
            dataset['val_df'], dataset['task'], dataset['num_labels'])

        train_pred_dfs.append(train_pred_df)
        train_scores.append(train_score)
        val_pred_dfs.append(val_pred_df)
        val_scores.append(val_score)

    df = pandas.DataFrame(grid)
    df['train_score'] = train_scores
    df['val_score'] = val_scores

    df_str = df.to_string(formatters={
        'l1_weight': lambda x: '%.1e' % x,
        'l2_weight': lambda x: '%.1e' % x,
        'train_score': lambda x: '%.3f' % x,
        'val_score': lambda x: '%.3f' % x})
    print(df_str)
    with open(out_dirnames['val'] + '/_results.txt', 'w') as f:
        f.write(df_str)

    best_ind = df['val_score'].argmax()
    best_score = df['val_score'].max()
    best_setting = dict(
        df.iloc[best_ind][
            ['l1_weight', 'l2_weight', 'num_passes', 'loss', 'quadratic']
        ]
    )
    print('Best setting: {}'.format(best_setting))
    print('Best score: {:.3f}'.format(best_score))

    best_train_pred_df = train_pred_dfs[best_ind]
    best_val_pred_df = val_pred_dfs[best_ind]

    return best_setting, best_score, best_train_pred_df, best_val_pred_df


def _get_preds_and_score(
        pred_filename, dataset_df, task, num_labels, loss_function='logistic'):
    """
    # TODO: actually pass loss_function to this function.

    Parameters
    ----------
    pred_filename: string
        Filename containing VW-formatted prediction scores for ids.
    dataset_df: pandas.DataFrame
        Contains 'label' column containing true values for ids.

    """
    # Read the prediction file.
    try:
        pred_df = pandas.read_csv(
            pred_filename, sep=' ', index_col=1, header=None, names=['pred'])
    except Exception as e:
        raise Exception("Could not read predictions: {}".format(e))
    pred_df.index = pred_df.index.astype(str)

    # If using logisitic loss, convert to [-1, 1].
    if loss_function == 'logistic':
        pred_df['pred'] = (2. / (1. + np.exp(-pred_df['pred'])) - 1.)

    # Set the true values as a column.
    pred_df['label'] = dataset_df['label']

    if task == 'clf' and num_labels == 2:
        metrics = vislab.results.binary_metrics(
            pred_df, name='', balanced=False,
            with_plot=False, with_print=False)
        return pred_df, metrics['accuracy']
    elif task == 'clf' and num_labels > 2:
        metrics = vislab.results.multiclass_metrics(
            pred_df, name='', balanced=False,
            with_plot=False, with_print=False)
        return pred_df, metrics['accuracy']
    elif task == 'regr':
        metrics = vislab.results.regression_metrics(
            pred_df, name='', balanced=False,
            with_plot=False, with_print=False)
        return pred_df, metrics['mse']
    else:
        raise Exception("Unknown task/num_labels combo: {}/{}".format(
            task, num_labels))
