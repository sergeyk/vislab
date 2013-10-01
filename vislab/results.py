"""
Code for parsing results of prediction tasks.
"""
import sklearn.metrics
import numpy as np
import pandas as pd


def classification_metrics(pred_df):
    """
    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must have columns 'pred' and 'true', which contain the predicted
        and true values.
        The predicted values can be real-valued; > 0 values are
        considered positive.
    """
    # The prediction we loaded is actually a raw score.
    pred_df['score'] = pred_df['pred']

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

    # TODO: add more metrics here

    metrics = {
        'score': score
    }
    return metrics


def regression_metrics(pred_df):
    r2_score = sklearn.metrics.r2_score(
        pred_df['label'].values, pred_df['pred'].values)
    metrics = {
        'r2_score': r2_score
    }
    return metrics
