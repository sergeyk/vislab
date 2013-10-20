import logging
import unittest
import pandas as pd
import numpy as np
import gzip
import os
import shutil
import test_context
import vislab.vw3


class TestVW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dirname = test_context.temp_dirname + '/test_vw'
        if os.path.exists(cls.temp_dirname):
            shutil.rmtree(cls.temp_dirname)
        vislab.util.makedirs(cls.temp_dirname)

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
        label_df_filename = test_context.support_dirname + '/simple/label_df.h5'

        output_dirname = vislab.util.makedirs(
            self.temp_dirname + '/cache_data')
        vislab.vw3._cache_data(
            label_df_filename, feat_filenames, output_dirname,
            bit_precision=18, verbose=False, force=False)
        assert(os.path.exists(output_dirname + '/cache.vw'))
        expected = """\
1 1.000000 0|first 0:-0.885972 1:-2.772593 |second 0:0.059139
1 1.000000 1|first 0:-1.376205 1:-0.390953 |second 0:0.857275
-1 1.000000 2|first 0:-0.160053 1:0.141953 |second 0:1.067757
-1 1.000000 3|first 0:-1.053145 1:0.521065 |second 0:-0.281805
1 1.000000 4|first 0:0.548937 1:0.0974 |second 0:-0.331867
1 1.000000 5|first 0:0.762685 1:0.523891 |second 0:0.680036
-1 1.000000 6|first 0:-0.298776 1:-1.676004 |second 0:1.156621
-1 1.000000 7|first 0:-0.896211 1:-0.345982 |second 0:-0.309105
1 1.000000 8|first 0:0.09446 1:0.390093 |second 0:-0.824078
-1 1.000000 9|first 0:-0.829799 1:-0.466419 |second 0:0.064953
"""
        with open(output_dirname + '/cache_preview.txt') as f:
            actual = f.read()
        assert(expected == actual)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
