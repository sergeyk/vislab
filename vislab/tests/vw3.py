import logging
import unittest
import pandas as pd
import numpy as np
import gzip
import os
import test_context
import vislab.predict
import vislab.vw3


class TestVW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dirname = vislab.util.cleardirs(
            test_context.temp_dirname + '/test_vw')

    def test_write_data_in_vw_format_float(self):
        feat_df = pd.DataFrame(
            data=[
                np.array([3.24, 5.666, 1., 0.0000001, 0.]),
                np.array([1.00000003, 5, 2, 0.001, -0.000001]),
            ],
            index=['loller', 'taco']
        )
        feat_name = 'best'
        assert(len(feat_df.columns) > 1)

        expected = """\
 idloller |best 0:3.24 1:5.666 2:1.0
 idtaco |best 0:1.0 1:5.0 2:2.0 3:0.001
"""

        output_filename = self.temp_dirname + \
            '/test_write_data_in_vw_format.txt'
        try:
            os.remove(output_filename)
        except:
            pass
        vislab.vw3.write_data_in_vw_format(feat_df, feat_name, output_filename)

        with open(output_filename) as f:
            actual = f.read()
        assert(expected == actual)

        # Try writing to gzipped file
        output_filename = self.temp_dirname + \
            '/test_write_data_in_vw_format.txt.gz'
        try:
            os.remove(output_filename)
        except:
            pass
        vislab.vw3.write_data_in_vw_format(feat_df, feat_name, output_filename)

        with gzip.open(output_filename) as f:
            actual = f.read()
        assert(expected == actual)

    def test_write_data_in_vw_format_single_column(self):
        feat_df = pd.DataFrame(
            data=[
                (np.array([2.0003, 2]),),
                (np.array([True, False, True, False, False, True]),)
            ],
            index=['id', 'badman']
        )
        feat_name = 'best'
        assert(len(feat_df.columns) == 1)

        expected = """\
 idid |best 0:2.0003 1:2.0
 idbadman |best 0 2 5
"""

        output_filename = self.temp_dirname + \
            '/test_write_data_in_vw_format_single_column.txt'
        try:
            os.remove(output_filename)
        except:
            pass
        vislab.vw3.write_data_in_vw_format(feat_df, feat_name, output_filename)

        with open(output_filename) as f:
            actual = f.read()
        assert(expected == actual)

    def test__cache_data(self):
        # These test file were created from the 'classifier tests' notebook.
        feat_filenames = [
            test_context.support_dirname + '/simple/first.txt',
            test_context.support_dirname + '/simple/second.txt.gz'
        ]
        label_df_filename = test_context.support_dirname + \
            '/simple/label_df.h5'

        output_dirname = vislab.util.makedirs(
            self.temp_dirname + '/cache_data')
        cache_cmd, preview_cmd = vislab.vw3._cache_cmd(
            label_df_filename, feat_filenames, output_dirname,
            2, bit_precision=18, verbose=False, force=False)
        vislab.util.run_through_bash_script(
            [cache_cmd, preview_cmd], None, verbose=False)

        assert(os.path.exists(output_dirname + '/cache.vw'))
        expected = """\
-1 1.000000 0|first 0:0.907699 1:0.910662 |second 0:1.057998
-1 1.000000 1|first 0:-0.375222 1:2.900907 |second 0:0.831044
-1 1.000000 2|first 0:-0.276823 1:1.717314 |second 0:-0.345345
-1 1.000000 3|first 0:0.596906 1:1.522828 |second 0:-0.766781
-1 1.000000 4|first 0:0.540094 1:0.094393 |second 0:-0.919987
1 1.000000 5|first 0:-0.972403 1:2.213648 |second 0:-0.0831
-1 1.000000 6|first 0:0.098378 1:0.200471 |second 0:-0.9833
1 1.000000 7|first 0:-0.755463 1:2.802532 |second 0:-0.642245
1 1.000000 8|first 0:-0.326318 1:0.74197 |second 0:1.21393
1 1.000000 9|first 0:-2.115056 1:0.353851 |second 0:1.62912
"""
        with open(output_dirname + '/cache_preview.txt') as f:
            actual = f.read()
        assert(expected == actual)

    def test__get_feat_filenames(self):
        feat_names = ['first', 'second']
        feat_dirname = test_context.support_dirname + '/simple'
        vislab.vw3._get_feat_filenames(feat_names, feat_dirname)

    def test_vw_fit_simple(self):
        label_df_filename = test_context.support_dirname + \
            '/simple/label_df.h5'
        label_df = pd.read_hdf(label_df_filename, 'df')
        dataset = vislab.predict.get_binary_or_regression_dataset(
            label_df, 'simple', 'label')

        feat_dirname = test_context.support_dirname + '/simple'
        vw = vislab.vw3.VW(self.temp_dirname + '/vw_simple')

        feat_names = ['first']
        pred_df, test_score, val_score, train_score = vw.fit_and_predict(
            dataset, feat_names, feat_dirname)
        print(feat_names, test_score, val_score, train_score)
        #assert(test_score > 0.7 and test_score < 0.8)

        feat_names = ['second']
        pred_df, test_score, val_score, train_score = vw.fit_and_predict(
            dataset, feat_names, feat_dirname)
        print(feat_names, test_score, val_score, train_score)
        #assert(test_score > 0.9)

        feat_names = ['first', 'second']
        pred_df, test_score, val_score, train_score = vw.fit_and_predict(
            dataset, feat_names, feat_dirname)
        print(feat_names, test_score, val_score, train_score)
        #assert(test_score > 0.9)

    def test_vw_fit_iris(self):
        label_df_filename = test_context.support_dirname + \
            '/iris/label_df.h5'
        label_df = pd.read_hdf(label_df_filename, 'df')

        dataset = vislab.predict.get_multiclass_dataset(
            label_df, 'iris', 'labels', ['label_0', 'label_1', 'label_2'])

        feat_dirname = test_context.support_dirname + '/iris'
        vw = vislab.vw3.VW(self.temp_dirname + '/vw_iris', num_passes=[10, 50, 100])

        feat_names = ['all']
        pred_df, test_score, val_score, train_score = vw.fit_and_predict(
            dataset, feat_names, feat_dirname)
        print(feat_names, test_score, val_score, train_score)
        assert(test_score > 0.8)
        # TODO: really want > .9 accuracy!


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
