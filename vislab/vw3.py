import numpy as np
import subprocess
import time
import logging
import os
import pandas as pd
import sklearn.grid_search
import vislab.util

vw_cmd = "vw --quiet --compressed"


def _feat_for_vw(id_, feat_name, feat, decimals=6):
    """
    Parameters
    ----------
    id_: string
        Identifying this example.
    feat_name: string
        To be used as the vw namespace.
    feat: ndarray
        Must be of either bool, int, or float dtype.
    decimals: int [6]
        Values will be rounded to this number of decimals.
    """
    if feat.dtype == 'bool':
        s = ' '.join(str(x) for x in np.where(feat)[0])

    elif feat.dtype in [float, int]:
        # Round to the specified number of decimals.
        feat = np.around(feat, decimals)

        # Set things that are almost 0 to 0.
        tolerance = 10**-decimals
        feat[np.abs(feat) <= tolerance] = 0

        s = ' '.join(
            '{}:{}'.format(i, x)
            for i, x in zip(np.arange(feat.size)[feat != 0], feat[feat != 0])
        )

    else:
        print feat
        raise ValueError('Unsupported feature dtype.')

    return ' id{} |{} {}'.format(id_, feat_name, s)


def write_data_in_vw_format(feat_df, feat_name, output_filename):
    """
    Parameters
    ----------
    feat_df: pandas.DataFrame
        The indices will be written out in the data file.
        The DataFrame should either:
        - contain one ndarray per row (1 column).
        - or have the feature values in the columns (> 1 columns).
    feat_name: string
        To use as the VW namespace for this feature.
    output_filename: string
        Make sure it's in a writeable directory.
        If it ends in '.gz', then the file will be gzipped.
    """
    fn_name = 'write_data_in_vw_format'

    # Establish what form the features are in.
    ndarray_feat = len(feat_df.columns) == 1

    # If the filename ends in gzip, we will output to the filename
    # without the '.gz', because the gzip command at the end of this
    # function will modify the output file in place and append '.gz'.
    gzip = False
    if output_filename.endswith('.gz'):
        output_filename = output_filename[:-3]
        gzip = True

    t = time.time()
    feat_lines = []
    for ix, row in feat_df.iterrows():
        if ndarray_feat:
            feat = row.values[0]
            # Check if there is only a single value here, and wrap if so
            try:
                len(feat)
            except:
                feat = np.array([feat])
        else:
            feat = row.values
        feat_lines.append(_feat_for_vw(ix, feat_name, feat))
    logging.info('{}: forming lines took {:.3f} s'.format(
        fn_name, time.time() - t))

    t = time.time()
    with open(output_filename, 'w') as f:
        f.write('\n'.join(feat_lines) + '\n')
    logging.info('{}: writing file took {:.3f} s'.format(
        fn_name, time.time() - t))

    if gzip:
        t = time.time()
        retcode = subprocess.call(['gzip', '-f', output_filename])
        assert(retcode == 0)
        logging.info('{}: gzip took {:.3f} s'.format(fn_name, time.time() - t))


def _cache_cmd(
        label_df_filename, feat_filenames, output_dirname,
        bit_precision=18, verbose=False, force=False):
    """
    Run the labels and feature data through VW once to output to cache.

    Parameters
    ----------
    label_df_filename: string
        A DataFrame in pickle or HDF5 format whose index is image ids.
    feat_filenames: sequence of string
        These features will be horizontally concatenated, so they should
        have different namespaces.
        These files must end in '.txt' or '.gz'.
    output_dirname: string
    bit_precision: int [18]
        Number of bits used in the hashing function.
        If the product of features and classes (for OAA mode) is greater
        than 260K, should increase from the default of 18.
    verbose: bool [False]
    force: bool [False]
    """
    assert(os.path.exists(label_df_filename))

    cache_filename = output_dirname + '/cache.vw'
    cache_preview_filename = output_dirname + '/cache_preview.txt'
    if not force and os.path.exists(cache_filename):
        print("Cache file exists, not doing anything.")
        return None, None

    # Concatenate the feature files horizontally.
    for f in feat_filenames:
        assert(os.path.exists(f))
    cats = [
        '<(zcat {})'.format(f) if f.endswith('.gz') else '<(cat {})'.format(f)
        for f in feat_filenames
    ]
    paste_cmd = "paste -d'\\0' {}".format(' '.join(cats))

    # The output of the above is piped through a filter that selects by id.
    vw_filter_filename = vislab.repo_dirname + '/vw_filter.py'
    filter_cmd = "python {} {}".format(vw_filter_filename, label_df_filename)

    # Output a few lines of what is piped into VW to a preview file.
    head_cmd = 'head -n 10'
    cache_preview_cmd = "{} | {} | {} > {}".format(
        paste_cmd, filter_cmd, head_cmd, cache_preview_filename)

    # Run all data through VW to cache it.
    cache_cmd = "{} | {} | {}".format(
        paste_cmd, filter_cmd, vw_cmd)
    cache_cmd += " -k --cache_file {} --bit_precision {} --noop".format(
        cache_filename, bit_precision)

    return cache_cmd, cache_preview_cmd


def _get_feat_filenames(feat_names, feat_dirname):
    """
    Get actual feature filenames and confirm they exist.
    """
    feat_filenames = []
    for feat_name in feat_names:
        normal_name = feat_dirname + '/' + feat_name + '.txt'
        gz_name = normal_name + '.gz'
        if os.path.exists(gz_name):
            feat_filenames.append(gz_name)
        elif os.path.exists(normal_name):
            feat_filenames.append(normal_name)
        else:
            raise Exception("Feature filename not found for '{}'".format(
                feat_name))
    return feat_filenames


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
        dirname, _setting_to_name(setting))

    # The basic command.
    cmd = vw_cmd + " --cache_file={}/cache.vw -f {}".format(
        dirname, model_filename)
    cmd += " --l1={} --l2={} --passes={} --loss_function={}".format(
        setting['l1'], setting['l2'], setting['num_passes'], setting['loss'])

    # If we are training from scratch, then we will use all data.
    if from_model is None:
        cmd += " --holdout_off --save_resume"

    # If we are training with the val data starting with the best model,
    # then we turn on early termination and set the holdout fraction to
    # 1/6, so that we don't overfit.
    else:
        cmd += " --early_terminate=3 --holdout_period=6 -i {}".format(
            from_model)

    return cmd


def _pred_vw_cmd(setting, model_dirname, pred_dirname):
    """
    Return command to predict with a VW model.
    It is important to output raw predictions here.

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
    name = _setting_to_name(setting)
    model_filename = '{}/{}_model.vw'.format(model_dirname, name)
    assert(os.path.exists(model_filename))
    pred_filename = '{}/{}_pred.txt'.format(pred_dirname, name)
    cmd = vw_cmd + " -t -i {} --cache_file={}/cache.vw -r {}".format(
        model_filename, pred_dirname, pred_filename)
    return cmd


def _setting_to_name(setting):
    return '_'.join(
        '{}_{}'.format(k, v)
        for k, v in sorted(setting.iteritems())
    )


def _train_with_val(
        dataset, output_dirnames, settings, num_workers=1, verbose=False):
    #train_df = dataset['train_df']
    val_df = dataset['val_df']
    test_df = dataset['test_df']
    num_labels = dataset['num_labels']
    train_dir = output_dirnames['train']
    val_dir = output_dirnames['val']
    test_dir = output_dirnames['test']

    print("Running VW training for {} param settings, {} at a time".format(
        len(settings), num_workers))

    # Run training commands in parallel.
    # Each VW uses at most two cores and very little memory, and so
    # can run in parallel with other instances. Since the instances
    # read data at roughly the same rate, OS caching should work.
    vislab.util.run_through_bash_script(
        [_train_vw_cmd(setting, train_dir) for setting in settings],
        train_dir + '/_train_cmds.sh',
        False, num_workers
    )

    # Run prediction commands on the validation data in parallel.
    vislab.util.run_through_bash_script(
        [_pred_vw_cmd(setting, train_dir, val_dir) for setting in settings],
        val_dir + '/_val_pred_cmds.sh',
        False, num_workers
    )

    # Run prediction commands on the test data in parallel.
    vislab.util.run_through_bash_script(
        [_pred_vw_cmd(setting, train_dir, test_dir) for setting in settings],
        test_dir + '/_test_pred_cmds.sh',
        False, num_workers
    )

    # Load all the model predictions and pick the best settings.
    val_scores = []
    test_scores = []
    for setting in settings:
        _, val_score = _score_predict(setting, val_dir, num_labels, val_df)
        val_scores.append(val_score)
        _, test_score = _score_predict(setting, test_dir, num_labels, test_df)
        test_scores.append(test_score)

        # test_pred_df = _read_preds(
        #     '{}/{}_pred.txt'.format(test_dir, _setting_to_name(setting)),
        #     num_labels, setting['loss'])
        # test_scores.append(_score_preds(test_pred_df, num_labels))

    df = pd.DataFrame(settings)
    df['val_score'] = val_scores
    df['test_score'] = test_scores

    df_str = df.to_string(
        formatters={
            'val_score': lambda x: '%.3f' % x,
            'test_score': lambda x: '%.3f' % x
        }
    )
    print(df_str)
    with open(val_dir + '/_results.txt', 'w') as f:
        f.write(df_str)

    best_ind = df['val_score'].argmax()
    best_score = df['val_score'].max()
    del df['val_score']
    del df['test_score']
    best_setting = dict(df.iloc[best_ind])

    print('Best setting: {}'.format(best_setting))
    print('Best score: {:.3f}'.format(best_score))

    return best_setting


def cache_files(
        dataset, feat_names, feat_dirname, dirname, bit_precision,
        num_workers, force=False):
    """
    Cache files for splits in the dataset.
    """
    # Get actual feature filenames.
    feat_filenames = _get_feat_filenames(feat_names, feat_dirname)

    split_names = ['train', 'val', 'test']
    output_dirnames = {}
    cache_cmds = []
    cache_preview_cmds = []
    for split_name in split_names:
        cache_dirname = '_'.join(feat_names)
        output_dirnames[split_name] = vislab.util.makedirs(
            '{}/{}/{}'.format(dirname, cache_dirname, split_name))

        # Save the dataframe with labels for use in filtering examples.
        df_filename = dirname + '/{}_df.h5'.format(split_name)
        if force or not os.path.exists(df_filename):
            dataset[split_name + '_df'].to_hdf(
                df_filename, 'df', mode='w')
        else:
            logging.info("Not writing out DataFrame for {}".format(
                split_name))

        # Cache data by running it through a filter.
        cache_cmd, cache_preview_cmd = _cache_cmd(
            df_filename, feat_filenames, output_dirnames[split_name],
            bit_precision, verbose=False, force=force)
        if cache_cmd is not None:
            cache_cmds.append(cache_cmd)
        if cache_preview_cmd is not None:
            cache_preview_cmds.append(cache_preview_cmd)

    logging.info("Caching data")
    t = time.time()
    vislab.util.run_through_bash_script(
        cache_preview_cmds, dirname + '/_cache_preview_cmds.sh',
        verbose=False, num_workers=num_workers)
    vislab.util.run_through_bash_script(
        cache_cmds, dirname + '/_cache_cmds.sh',
        verbose=False, num_workers=num_workers)
    logging.info('Caching data took {:.3f} s'.format(time.time() - t))

    return output_dirnames


def _predict(setting, source_dirname, target_dirname, gt_df, num_labels):
    pred_cmd = _pred_vw_cmd(
        setting, source_dirname, target_dirname)
    vislab.util.run_through_bash_script(
        [pred_cmd], target_dirname + '/_test_cmd.sh', verbose=False)
    return _score_predict(setting, target_dirname, num_labels, gt_df)


def _score_predict(setting, dirname, num_labels, gt_df):
    pred_filename = '{}/{}_pred.txt'.format(dirname, _setting_to_name(setting))
    pred_df = _read_preds(pred_filename, num_labels, setting['loss'])
    pred_df = pred_df.join(gt_df)
    score = _score_preds(pred_df, num_labels)
    return pred_df, score


def _read_preds(pred_filename, num_labels, loss_function):
    try:
        pred_df = pd.read_csv(
            pred_filename, sep=' ', index_col=1, header=None, names=['pred'])
    except Exception as e:
        raise Exception("Could not read predictions: {}".format(e))
    pred_df.index = pred_df.index.astype(str)

    # If using logisitic loss, convert to [-1, 1].
    if loss_function == 'logistic':
        pred_df['pred'] = (2. / (1. + np.exp(-pred_df['pred'])) - 1.)

    return pred_df


def _score_preds(pred_df, num_labels):
    """
    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must have both 'pred' and 'label' columns.
    """
    if num_labels < 0:
        metrics = vislab.results.regression_metrics(
            pred_df, name='', balanced=False,
            with_plot=False, with_print=False)
        return metrics['mse']

    elif num_labels == 2:
        metrics = vislab.results.binary_metrics(
            pred_df, name='', balanced=True,
            with_plot=False, with_print=False)
        return metrics['accuracy']

    elif num_labels > 2:
        metrics = vislab.results.multiclass_metrics(
            pred_df, name='', balanced=False,
            with_plot=False, with_print=False)
        # TODO: does this metric make sense for cross-validation?
        return metrics['binary_metrics_df'].ix['_mean']['ap']

    else:
        raise ValueError("Illegal num_labels")


class VW(object):
    """
    Initialize with a dirname and three DataFrames (train, val, test),
    each containing a 'label' column and an 'importance' column.
    Additionally, provide sequences of parameters to do grid search over
    during training.

    TODO:
    - take collection_name: store results into mongodb
    - can calculate importance here, instead of relying on the dataframe
    """
    def __init__(
            self, dirname, num_workers=6, bit_precision=18,
            num_passes=[25],
            loss=['hinge', 'logistic'],
            l1=['0', '1e-6', '1e-9'],
            l2=['0', '1e-6', '1e-9']):
        # Actual output directory will have bit_precision info in name,
        # because cache files are dependent on the precision.
        self.dirname = vislab.util.makedirs(
            dirname + '_b{}'.format(bit_precision))
        self.bit_precision = bit_precision
        self.num_workers = num_workers

        # Set the parameter grid, ordering it such that same number of
        # passes are ordered together, for more efficient parallelism.
        self.param_grid = {
            'loss': loss,
            'num_passes': num_passes,
            'l1': [str(x) for x in l1],
            'l2': [str(x) for x in l2]
        }
        settings_df = pd.DataFrame(
            list(sklearn.grid_search.ParameterGrid(self.param_grid))
        ).sort(['num_passes', 'loss'])
        self.settings = [dict(row) for ind, row in settings_df.iterrows()]

    def fit_and_predict(self, dataset, feat_names, feat_dirname, force=False):
        """
        Parameters
        ----------
        feat_names: sequence of string
        feat_dirname: string
            Directory containing VW-format files for the feat_names.
            Each file contains features in VW format, and can be gzippd.
            Use write_data_in_vw_format() to output these.
        dataset: dict
            Contains 'train_df', 'val_df', 'test_df' label DataFrames,
            and 'num_labels': an int.
        """
        # Cache all splits to VW format.
        output_dirnames = cache_files(
            dataset, feat_names, feat_dirname, self.dirname,
            self.bit_precision, self.num_workers, force)

        # Train models in a grid search over params.
        best_setting = _train_with_val(
            dataset, output_dirnames, self.settings,
            self.num_workers, verbose=False)

        # Update the best model with validation data.
        print("Updating best VW model with validation data")
        best_model_filename = '{}/{}_model.vw'.format(
            output_dirnames['train'], _setting_to_name(best_setting))
        cmd = _train_vw_cmd(
            best_setting, output_dirnames['val'], best_model_filename)
        cmd_filename = output_dirnames['val'] + '/_train_cmd.sh'
        vislab.util.run_through_bash_script(
            [cmd], cmd_filename, verbose=False)

        # Get predictions from all the splits. Need to predict on train.
        print("Running VW prediction on all splits.")
        train_pred_df, train_score = _predict(
            best_setting, output_dirnames['val'], output_dirnames['train'],
            dataset['train_df'], dataset['num_labels'])
        val_pred_df, val_score = _predict(
            best_setting, output_dirnames['val'], output_dirnames['val'],
            dataset['val_df'], dataset['num_labels'])
        test_pred_df, test_score = _predict(
            best_setting, output_dirnames['val'], output_dirnames['test'],
            dataset['test_df'], dataset['num_labels'])

        # TODO: keeping this around in case retraining starts messing up
        # train_pred_df, train_score = _predict(
        #     best_setting, output_dirnames['train'], output_dirnames['train'],
        #     dataset['train_df'], dataset['num_labels'])
        # val_pred_df, val_score = _score_predict(
        #     best_setting, output_dirnames['val'],
        #     dataset['num_labels'], dataset['val_df'])
        # test_pred_df, test_score = _score_predict(
        #     best_setting, output_dirnames['test'],
        #     dataset['num_labels'], dataset['test_df'])

        # Combine all predictions into one DataFrame.
        train_pred_df['split'] = 'train'
        val_pred_df['split'] = 'val'
        test_pred_df['split'] = 'test'
        pred_df = train_pred_df.append(val_pred_df).append(test_pred_df)

        return pred_df, test_score, val_score, train_score
