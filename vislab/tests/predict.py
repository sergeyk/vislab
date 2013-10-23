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
        # This test just checks for success.
        assert(dataset is not None)

    def test_get_multiclass_dataset_no_multilabel(self):
        label_df = pd.DataFrame(
            {
                'style_Abstract': [
                    True, True, False, False, False, False, False, True
                ],
                'style_Scary': [
                    False, False, False, True, False, False, True, False
                ],
                'style_Doglike': [
                    False, False, True, False, True, True, False, False
                ],
            },
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        )
        dataset_name = 'boogaloo'
        column_names = ['style_Abstract', 'style_Scary', 'style_Doglike']
        column_set_name = 'styles'

        expected = {
            'column_names': [
                'style_Abstract', 'style_Scary', 'style_Doglike'
            ],
            'dataset_name': 'boogaloo',
            'name': 'boogaloo_styles_train_4',
            'num_labels': 3,
            'salient_parts': {
                'data': 'boogaloo_styles',
                'num_test': 2,
                'num_train': 4,
                'num_val': 2
            },
            'task': 'clf',
            'test_df': pd.DataFrame(
                {
                    'label': [1, 1],
                    'importance': [1, 1],
                    'style_Abstract': [True, False],
                    'style_Scary': [False, False],
                    'style_Doglike': [False, True],
                },
                ['two', 'six']
            ),
            'val_df': pd.DataFrame(
                {
                    'label': [3, 1],
                    'importance': [1, 1],
                    'style_Abstract': [False, True],
                    'style_Scary': [False, False],
                    'style_Doglike': [True, False],
                },
                ['three', 'eight']
            ),
            'train_df': pd.DataFrame(
                {
                    'label': [1, 2, 2, 3],
                    'importance': [2, 1, 1, 2],
                    'style_Abstract': [True, False, False, False],
                    'style_Scary': [False, True, True, False],
                    'style_Doglike': [False, False, False, True],
                },
                ['one', 'seven', 'four', 'five']
            )
        }
        actual = vislab.predict.get_multiclass_dataset(
            label_df, dataset_name, column_set_name, column_names,
            test_frac=.2, balanced=False, random_seed=42)
        vislab.tests.util.assert_dicts_equal(expected, actual)

    def test_get_multiclass_dataset_multilabel(self):
        label_df = pd.DataFrame(
            {
                'style_Abstract': [
                    True, True, False, False, False, False, False, True
                ],
                'style_Scary': [
                    True, False, False, True, True, False, True, False
                ],
                'style_Doglike': [
                    False, True, True, False, True, True, False, False
                ],
            },
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        )
        dataset_name = 'boogaloo'
        column_names = ['style_Abstract', 'style_Scary', 'style_Doglike']
        column_set_name = 'styles'

        expected = {
            'column_names': [
                'style_Abstract', 'style_Scary', 'style_Doglike'
            ],
            'dataset_name': 'boogaloo',
            'name': 'boogaloo_styles_train_3',
            'num_labels': 3,
            'salient_parts': {
                'data': 'boogaloo_styles',
                'num_test': 3,
                'num_train': 3,
                'num_val': 2
            },
            'task': 'clf',
            'test_df': pd.DataFrame(
                {
                    'label': [1, 1, 1],
                    'importance': [1, 1, 1],
                    'style_Abstract': [True, True, False],
                    'style_Scary': [True, False, True],
                    'style_Doglike': [False, True, True],
                },
                ['one', 'two', 'five']
            ),
            'val_df': pd.DataFrame(
                {
                    'label': [2, 3],
                    'importance': [1, 1],
                    'style_Abstract': [False, False],
                    'style_Scary': [True, False],
                    'style_Doglike': [False, True],
                },
                ['four', 'three']
            ),
            'train_df': pd.DataFrame(
                {
                    'label': [2, 3, 1],
                    'importance': [1, 1, 1],
                    'style_Abstract': [False, False, True],
                    'style_Scary': [True, False, False],
                    'style_Doglike': [False, True, False],
                },
                ['seven', 'six', 'eight']
            )
        }
        actual = vislab.predict.get_multiclass_dataset(
            label_df, dataset_name, column_set_name, column_names,
            test_frac=.2, balanced=False, random_seed=42)
        vislab.tests.util.assert_dicts_equal(expected, actual)

    def test_get_multiclass_dataset_multilabel_with_split(self):
        label_df = pd.DataFrame(
            {
                'style_Abstract': [
                    True, True, False, False, False, False, False, True
                ],
                'style_Scary': [
                    True, False, False, True, True, False, True, False
                ],
                'style_Doglike': [
                    False, True, True, False, True, True, False, False
                ],
                '_split': [
                    'test', 'test', 'train', 'train',
                    'test', 'test', 'train', 'train'
                ]
            },
            ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
        )
        dataset_name = 'boogaloo'
        column_names = ['style_Abstract', 'style_Scary', 'style_Doglike']
        column_set_name = 'styles'

        expected = {
            'column_names': [
                'style_Abstract', 'style_Scary', 'style_Doglike'
            ],
            'dataset_name': 'boogaloo',
            'name': 'boogaloo_styles_train_2',
            'num_labels': 3,
            'salient_parts': {
                'data': 'boogaloo_styles',
                'num_test': 4,
                'num_train': 2,
                'num_val': 2
            },
            'task': 'clf',
            'test_df': pd.DataFrame(
                {
                    'label': [1, 1, 1, 1],
                    'importance': [1, 1, 1, 1],
                    'style_Abstract': [True, True, False, False],
                    'style_Scary': [True, False, True, False],
                    'style_Doglike': [False, True, True, True],
                },
                ['one', 'two', 'five', 'six']
            ),
            'val_df': pd.DataFrame(
                {
                    'label': [2, 3],
                    'importance': [1, 1],
                    'style_Abstract': [False, False],
                    'style_Scary': [True, False],
                    'style_Doglike': [False, True],
                },
                ['four', 'three']
            ),
            'train_df': pd.DataFrame(
                {
                    'label': [2, 1],
                    'importance': [1, 1],
                    'style_Abstract': [False, True],
                    'style_Scary': [True, False],
                    'style_Doglike': [False, False],
                },
                ['seven', 'eight']
            )
        }
        actual = vislab.predict.get_multiclass_dataset(
            label_df, dataset_name, column_set_name, column_names,
            test_frac=.2, balanced=False, random_seed=42)
        vislab.tests.util.assert_dicts_equal(expected, actual)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
