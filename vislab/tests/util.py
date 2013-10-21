import pandas as pd


def assert_dicts_equal(expected, actual):
    assert(set(expected.keys()) == set(actual.keys()))
    for key, expected_val in expected.iteritems():
        if isinstance(expected_val, pd.DataFrame):
            # order matters...
            df = actual[key].reindex(
                index=expected_val.index,
                columns=expected_val.columns
            )
            assert((expected_val == df).all().all())
