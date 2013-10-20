import numpy as np
import subprocess
import time
import logging
import os
import vislab.util


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


def _cache_data(
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
        return

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
    cache_cmd = "{} | {} | vw -k --cache_file {}".format(
        paste_cmd, filter_cmd, cache_filename)
    cache_cmd += " --bit_precision {} --noop --quiet --compressed".format(
        bit_precision)

    logging.info("Caching data")
    t = time.time()
    cmd_filename = output_dirname + '/_cache_cmd.sh'
    vislab.util.run_through_bash_script(
        [cache_preview_cmd, cache_cmd], cmd_filename, verbose)
    logging.info('Caching data took {:.3f} s'.format(time.time() - t))


def _get_feat_filenames(feat_names, feat_dirname):
    """
    Get actual feature filenames and confirm they exist.
    """
    feat_filenames = []
    for feat_name in feat_names:
        normal_name = feat_dirname + '/' + feat_name + '.txt.gz'
        gz_name = normal_name + '.gz'
        if os.path.exist(gz_name):
            feat_filenames.append(gz_name)
        elif os.path.exist(normal_name):
            feat_filenames.append(normal_name)
        else:
            raise Exception('Feature filename not found for {}'.format(
                feat_name))
    return feat_filenames


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
            self, dirname, num_workers=4, bit_precision=18,
            num_passes=[10], loss=['hinge'], l1=[0], l2=[0]):
        # Actual output directory will have bit_precision info in name,
        # because cache files are dependent on the precision.
        self.dirname = vislab.util.makedirs(dirname + '_b' + bit_precision)
        self.bit_precision = bit_precision
        self.num_passes = num_passes
        self.loss = loss
        self.l1 = l1
        self.l2 = l2

    def fit(self, label_df_dict, feat_names, feat_dirname, force=False):
        """
        Parameters
        ----------
        feat_names: sequence of string
        feat_dirname: string
            Directory containing VW-format files for the feat_names.
            Each file contains features in VW format, and can be gzippd.
            Use write_data_in_vw_format() to output these.
        label_df_dict: dict
            Contains 'train_df', 'val_df', 'test_df' label DataFrames.
        """
        # Get actual feature filenames.
        feat_filenames = _get_feat_filenames(feat_names, feat_dirname)

        # Cache all splits to VW format.
        split_names = ['train', 'val', 'test']
        for split_name in split_names:
            cache_dirname = '_'.join(feat_names)
            split_cache_dirname = vislab.util.makedirs(
                '{}/{}/{}'.format(self.dirname, cache_dirname, split_name))
            _cache_data(
                label_df_dict[split_name], feat_filenames, split_cache_dirname,
                self.bit_precision, verbose=False, force=force)
