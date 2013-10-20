import logging
import unittest
import pandas as pd
import numpy as np
import gzip
from test_context import *

import vislab.vw3


class TestBlob(unittest.TestCase):
    def setUp(self):
        pass

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

        output_filename = temp_dirname + '/test_write_data_in_vw_format.txt'
        try:
            os.remove(output_filename)
        except:
            pass
        vislab.vw3.write_data_in_vw_format(feat_df, feat_name, output_filename)

        with open(output_filename) as f:
            actual = f.read()
        assert(expected == actual)

        # Try writing to gzipped file
        output_filename = temp_dirname + '/test_write_data_in_vw_format.txt.gz'
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

        output_filename = temp_dirname + \
            '/test_write_data_in_vw_format_single_column.txt'
        try:
            os.remove(output_filename)
        except:
            pass
        vislab.vw3.write_data_in_vw_format(feat_df, feat_name, output_filename)

        with open(output_filename) as f:
            actual = f.read()
            print actual
        assert(expected == actual)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
