import numpy as np
import subprocess
import time
import logging
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
            feat = row[0]
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
        retcode = subprocess.call(['gzip', output_filename])
        assert(retcode == 0)
        logging.info('{}: gzip took {:.3f} s'.format(fn_name, time.time() - t))


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
            self, dirname, num_workers=4,
            num_passes=[10], loss=['hinge'], l1=[0], l2=[0]):
        self.dirname = vislab.util.makedirs(dirname)
        self.num_passes = num_passes
        self.loss = loss
        self.l1 = l1
        self.l2 = l2

    def fit(self, feat_filename, labels_df_filename):
        """
        Parameters
        ----------
        feat_filename: string
            Contains features
        """
        pass
