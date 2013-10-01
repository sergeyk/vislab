"""
Code for parsing results of prediction tasks.
"""
import sklearn.metrics
import numpy as np
import pandas as pd


def classification_report(pred_df, loss_function='logistic'):
    """
    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must have columns 'pred' and 'true', which contain the predicted
        and true values.
    """
    # The prediction we loaded is actually a raw score.
    pred_df['score'] = pred_df['pred']

    # If using logisitic loss, convert to [-1, 1].
    if loss_function == 'logistic':
        pred_df['score'] = (2. / (1. + np.exp(-pred_df['score'])) - 1.)

    # To compute accuracy, convert to binary predictions.
    pred_df['pred'] = -1
    pred_df['pred'][pred_df['score'] > 0] = 1

    # Geometric mean of the pos and neg accuracies is taken to be
    # robust to label imbalance.
    pos_ind = pred_df['label'] == 1

    y_true = pred_df['label'][pos_ind].values
    y_pred = pred_df['pred'][pos_ind].values
    pos_score = sklearn.metrics.accuracy_score(y_true, y_pred)

    y_true = pred_df['label'][~pos_ind].values
    y_pred = pred_df['pred'][~pos_ind].values
    neg_score = sklearn.metrics.accuracy_score(y_true, y_pred)

    score = np.sqrt(pos_score * neg_score)


def regression_report(pred_df, loss_function='squared'):
    r2_score = sklearn.metrics.r2_score(
        pred_df['label'].values, pred_df['pred'].values)
    return {
        'r2_score': r2_score
    }
