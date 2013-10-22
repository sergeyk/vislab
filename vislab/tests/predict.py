import logging
import pandas as pd
import unittest
import test_context
import vislab.predict
import vislab.tests.util


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dirname = vislab.util.cleardirs(
            test_context.temp_dirname + '/test_predict')

    def test_get_binary_dataset(self):
        label_df = pd.DataFrame(
            {
                'style_Abstract': [
                    True, True, False, False, True, True, False, True
                ],
                'style_Scary': [
                    False, False, True, True, True, True, False, False
                ]
            },
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        )
        dataset_name = 'boogaloo'
        column_name = 'style_Abstract'

        expected = {
            'dataset_name': 'boogaloo',
            'name': 'boogaloo_style_Abstract_train_4',
            'num_labels': 2,
            'salient_parts': {
                'data': 'boogaloo_style_Abstract',
                'num_test': 2,
                'num_train': 4,
                'num_val': 2
            },
            'task': 'clf',
            'train_df': pd.DataFrame(
                {
                    'label': [1, 1, 1, -1],
                    'importance': [1., 1., 1., 3.]
                },
                ['five', 'one', 'six', 'three']
            ),
            'val_df': pd.DataFrame(
                {
                    'label': [1, -1],
                    'importance': [1., 1.]
                },
                ['eight', 'seven']
            ),
            'test_df': pd.DataFrame(
                {
                    'label': [1, -1],
                    'importance': [1., 1.]
                },
                ['two', 'four']
            )
        }

        actual = vislab.predict.get_binary_or_regression_dataset(
            label_df, dataset_name, column_name, test_frac=.2, min_pos_frac=.1)

        vislab.tests.util.assert_dicts_equal(expected, actual)

    def test_get_regression_dataset(self):
        label_df = pd.DataFrame(
            {
                'style_Abstract': [
                    1., 2., 3., 4., 5., 6., 7., 8.
                ],
                'style_Scary': [
                    1., 2., 3., 4., 5., 6., 7., 8.
                ]
            },
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        )
        dataset_name = 'boogaloo'
        column_name = 'style_Abstract'

        expected = {
            'dataset_name': 'boogaloo',
            'name': 'boogaloo_style_Abstract_train_4',
            'num_labels': -1,
            'salient_parts': {
                'data': 'boogaloo_style_Abstract',
                'num_test': 2,
                'num_train': 4,
                'num_val': 2
            },
            'task': 'regr',
            'train_df': pd.DataFrame(
                {
                    'label': [3., 5., 4., 7.],
                    'importance': [1., 1., 1., 1.]
                },
                ['three', 'five', 'four', 'seven']
            ),
            'val_df': pd.DataFrame(
                {
                    'label': [2., 6.],
                    'importance': [1., 1.]
                },
                ['two', 'six']
            ),
            'test_df': pd.DataFrame(
                {
                    'label': [1., 8.],
                    'importance': [1., 1.]
                },
                ['one', 'eight']
            )
        }

        actual = vislab.predict.get_binary_or_regression_dataset(
            label_df, dataset_name, column_name, test_frac=.2, min_pos_frac=.1)

        vislab.tests.util.assert_dicts_equal(expected, actual)

    def test_get_binary_dataset_from_simple_label_df(self):
        label_df_filename = test_context.support_dirname + \
            '/simple/label_df.h5'
        label_df = pd.read_hdf(label_df_filename, 'df')
        dataset = vislab.predict.get_binary_or_regression_dataset(
            label_df, 'simple', 'label')

    @unittest.skip("not implemented yet")
    def test_get_multiclass_dataset(self):
        pass


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
