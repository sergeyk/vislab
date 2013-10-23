import numpy as np
import pandas as pd


def assert_dicts_equal(expected, actual):
    assert(set(expected.keys()) == set(actual.keys()))
    for key, expected_val in expected.iteritems():
        if isinstance(expected_val, pd.DataFrame):
            assert(np.all(actual[key].shape == expected_val.shape))
            # order matters...
            df = actual[key].reindex(
                index=expected_val.index,
                columns=expected_val.columns
            )
            assert(np.all(df.columns == expected_val.columns))
            assert((expected_val == df).all().all())
