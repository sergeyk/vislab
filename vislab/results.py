"""
Code for parsing results of prediction tasks.
"""
import sklearn.metrics
import numpy as np
import pandas as pd
import vislab


def compute_binary_metrics(df):
    """
    """
    metrics = {}
    results = sklearn.metrics.precision_recall_fscore_support(
        df['label'], df['pred_bin'])
    metrics['results_df'] = pd.DataFrame(
        np.array(results).T,
        columns=[['precision', 'recall', 'f1-score', 'support']],
        index=[['False', 'True']])
    metrics['mcc'] = sklearn.metrics.matthews_corrcoef(
        df['label'], df['pred_bin'])
    metrics['accuracy'] = sklearn.metrics.accuracy_score(
        df['label'], df['pred_bin'])
    pr_fig, prec, rec, metrics['ap'] = vislab.results_viz.plot_pr_curve(
        df['label'], df['pred'])
    roc_fig, fpr, tpr, metrics['auc'] = vislab.results_viz.plot_roc_curve(
        df['label'], df['pred'])
    return metrics


def print_binary_metrics(metrics, name):
    print('-'*60)
    if len(name) > 0:
        name = 'on the {} task'.format(name)
    print("Classification metrics {}".format(name))
    print(metrics['results_df'].to_string())
    print("Matthews' Correlation Coefficient: {}".format(metrics['mcc']))
    print("Accuracy: {}".format(metrics['accuracy']))
    print('')


def binary_metrics(pred_df, balanced=True):
    """
    Binary classification metrics.

    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must contain 'label' (int) and 'pred' (float) columns.
    balanced: bool [True]
        If True, the evaluation considers a class-balanced subset of
        the dataset.
    """
    # Drop rows without a label and make sure it's a bool.
    pred_df = pred_df.dropna(subset=['label'])
    pred_df['label'] = pred_df['label'] > 0

    # The prediction we loaded is a raw score: convert to binary prediction.
    pred_df['pred_bin'] = pred_df['pred'] > 0

    # Get a balanced subset if requested.
    if balanced:
        # Find the most frequent label
        counts = pred_df['label'].value_counts()
        mfl = counts.index[counts.argmax()]

        # Subsample to match the number of the other label.
        ind = np.random.permutation(counts.max())[:counts.min()]
        mfl_pred_df = pred_df[pred_df['label'] == mfl]
        lfl_pred_df = pred_df[~(pred_df['label'] == mfl)]
        pred_df = (mfl_pred_df.iloc[ind]).append(lfl_pred_df)

    metrics = compute_binary_metrics(pred_df)
    name = 'balanced dataset' if balanced else 'full dataset'
    print_binary_metrics(metrics, name)


def regression_metrics(pred_df):
    # TODO
    r2_score = sklearn.metrics.r2_score(
        pred_df['label'].values, pred_df['pred'].values)
    metrics = {
        'r2_score': r2_score
    }
    return metrics
