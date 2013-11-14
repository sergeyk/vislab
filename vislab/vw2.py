import numpy as np
import sklearn.preprocessing
import pandas as pd
import vislab


class VW(object):
    def __init__(
            self, dirname, bit_precision=18,
            loss_function=['hinge'], l1=0, l2=0):
        vislab.util.makedirs(dirname)
        self.model_filename = dirname + '/model.vw'
        self.data_filename = dirname + '/{}_data.txt'
        self.cache_filename = dirname + '/{}_cache.vw'
        self.pred_filename = dirname + '/{}_pred.txt'
        self.raw_pred_filename = dirname + '/{}_raw_pred.txt'
        self.quadratic = ''
        self.options = '--bit_precision={} --l1={} --l2={} --loss_function={} --holdout_off'.format(
            bit_precision, l1, l2, loss_function)

    def fit(self, X, y, num_passes=100, quadratic=''):
        """
        If y is boolean, does binary classification.
        If y is int or string-valued, does multilabel classification.
        """
        # Write out features and stream to VW cache.
        self._write_features(X, self.data_filename.format('train'), y)
        cmd = 'vw -d {} -k --cache_file {} --noop'.format(
            self.data_filename.format('train'),
            self.cache_filename.format('train')
        )
        if self.num_labels > 2:
            cmd += ' --oaa={}'.format(self.num_labels)
        stdout, stderr = vislab.util.run_shell_cmd(cmd, False)

        # Train several models, return the one with best validation performance.
        # param_grid = {
        #     'loss': loss,
        #     'num_passes': num_passes,
        #     'l1_weight': l1_weight,
        #     'l2_weight': l2_weight,
        #     'quadratic': [quadratic]
        # }
        # grid = list(sklearn.grid_search.ParameterGrid(param_grid))

        cmd = 'vw --cache_file {} -f {} --passes {} {}'.format(
            self.cache_filename.format('train'),
            self.model_filename,
            num_passes,
            self.options)
        if len(quadratic) > 0:
            cmd += ' --quadratic={}'.format(quadratic)
            self.quadratic = quadratic
        if self.num_labels > 2:
            cmd += ' --oaa={}'.format(self.num_labels)
        stdout, stderr = vislab.util.run_shell_cmd(cmd, False)

        return self

    def predict(self, X):
        y_pred, y_pred_bin = self._predict_with_proba(X)
        return y_pred_bin

    def predict_proba(self, X):
        y_pred, y_pred_bin = self._predict_with_proba(X)
        return y_pred

    def _predict_with_proba(self, X):
        self._write_features(X, self.data_filename.format('test'))

        cmd = 'vw -d {} -k -t --cache_file {} -i {} -p {} -r {}'.format(
            self.data_filename.format('test'),
            self.cache_filename.format('test'),
            self.model_filename,
            self.pred_filename.format('test'),
            self.raw_pred_filename.format('test'))
        if len(self.quadratic) > 0:
            cmd += ' --quadratic={}'.format(self.quadratic)
        stdout, stderr = vislab.util.run_shell_cmd(cmd, False)

        return self.read_preds('test')

    def read_preds(self, name='test'):
        # TODO: switch to actually reading the labels
        if self.num_labels > 2:
            df = pd.read_csv(self.raw_pred_filename.format('test'),
                             sep=' ', header=None)
            y_pred = df.apply(lambda x: [float(y.split(':')[1]) for y in x], raw=True).values
            y_pred_bin = y_pred.argmax(axis=1)
            # Note that we are not subtracting 1 from y_pred:
            # that is because the dataframe is again 0-indexed
            y_pred_bin = self.le.inverse_transform(y_pred_bin)

        else:
            y_pred = pd.read_csv(self.raw_pred_filename.format('test'),
                                 sep=' ', header=None)[0].values
            y_pred_bin = y_pred > 0

        return y_pred, y_pred_bin

    def _write_features(self, X, data_filename, y=None, shuffle=False):
        # Validate data.
        if y is not None:
            assert(X.shape[0] == y.shape[0])
            assert(y.dtype != float)

            # If doing binary classification, make sure we have both labels.
            if y.dtype == bool:
                assert(len(np.unique(y)) == 2)
                self.num_labels = 2
                y_ = -np.ones_like(y).astype(int)
                y_[y] = 1
                y = y_

            # If doing multiclass classification, make sure we have more than two labels,
            # and map the labels to ints on [1, K].
            else:
                num_labels = len(np.unique(y))
                assert(num_labels > 2)
                self.num_labels = num_labels

                self.le = sklearn.preprocessing.LabelEncoder()
                self.le.fit(y)
                y = self.le.transform(y) + 1

            # Permute features for well-behaved SGD.
            if shuffle:
                ind = np.random.permutation(X.shape[0])
                X = X[ind]
                y = y[ind]

        else:
            y = np.ones(X.shape[0])

        # Write out features to VW format.
        lines = []
        for row, label in zip(X, y):
            vals = ' '.join('{}:{}'.format(i, v) for i, v in enumerate(row) if v != 0)
            lines.append('{} |a {}'.format(label, vals))

        with open(data_filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')
