"""
Code for parsing results of prediction tasks.
"""
import sklearn.metrics
import numpy as np
import pandas as pd
import re
import vislab


def all_binary_metrics_df(
        preds_panel, label_df, selected_pred, pred_prefix, balanced=True):
    mc_pred_df = preds_panel.minor_xs(selected_pred).join(label_df)

    r = re.compile(pred_prefix)
    label_cols = [
        col.replace(pred_prefix + '_', '') for col in mc_pred_df.columns
        if r.match(col)
    ]

    all_metrics = {}
    for label in label_cols:
        pred_df = preds_panel['{}_{}'.format(pred_prefix, label)]
        pred_df['pred'] = pred_df[selected_pred]
        all_metrics[label] = vislab.results.binary_metrics(
            pred_df, 'name doesnt matter', balanced, with_plot=False)
    metrics_df = pd.DataFrame(all_metrics).T
    return metrics_df


def binary_metrics(
        pred_df, name='', balanced=True, with_plot=False, with_print=False):
    """
    Binary classification metrics.

    Parameters
    ----------
    pred_df: pandas.DataFrame
        Must contain 'label' (int) and 'pred' (float) columns.
    name: string ['']
        Name of the classification task: for example, the class name.
    balanced: bool [True]
        If True, the evaluation considers a class-balanced subset of
        the dataset.
    with_plot: bool [False]
        If True, plot curves and return handles to figures (otherwise
        return handles to None).
    with_print: bool [False]
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
        mfl = counts.index[0]

        # Subsample to match the number of the other label.
        ind = np.random.permutation(counts.max())[:counts.min()]
        mfl_pred_df = pred_df[pred_df['label'] == mfl]
        lfl_pred_df = pred_df[~(pred_df['label'] == mfl)]
        pred_df = (mfl_pred_df.iloc[ind]).append(lfl_pred_df)

    # Compute metrics.
    metrics = {}

    results = sklearn.metrics.precision_recall_fscore_support(
        pred_df['label'], pred_df['pred_bin'])
    metrics['results_df'] = pd.DataFrame(
        np.array(results).T,
        columns=[['precision', 'recall', 'f1-score', 'support']],
        index=[['False', 'True']])

    metrics['mcc'] = sklearn.metrics.matthews_corrcoef(
        pred_df['label'], pred_df['pred_bin'])

    metrics['accuracy'] = sklearn.metrics.accuracy_score(
        pred_df['label'], pred_df['pred_bin'])

    metrics['pr_fig'], prec, rec, metrics['ap'] = \
        get_pr_curve(pred_df['label'], pred_df['pred'], name, with_plot)

    metrics['roc_fig'], fpr, tpr, metrics['auc'] = \
        get_roc_curve(pred_df['label'], pred_df['pred'], name)

    if with_print:
        name = '{} balanced' if balanced else '{} full'
        print_metrics(metrics, name.format(name))

    return metrics


def multiclass_metrics(mc_pred_df, pred_prefix, balanced=True):
    """
    Multiclass classification metrics.

    Parameters
    ----------
    mc_pred_df: pandas.DataFrame
        Contains a 'label' ([1, K] int) column, and K
        '{}_{}'.format(pred_prefix, label) (float) columns,
        one for each label.
    pred_prefix: string
        The prefix before the name of the label column.
    balanced: bool [True]
        If True, the number if instances of each class is equalized.
    """
    metrics = {}

    # Get the list of labels.
    r = re.compile(pred_prefix)
    pred_cols, label_cols = map(list, zip(*[
        (col, col.replace(pred_prefix + '_', '')) for col in mc_pred_df.columns
        if r.match(col)
    ]))

    # Make two dataframes: for labels and predictions.
    # Drop those rows that don't have a single positive label.
    label_df = mc_pred_df[label_cols]
    ind = label_df.sum(1) > 0
    label_df = label_df[ind]
    pred_df = mc_pred_df[pred_cols][ind]

    # Get array of multi-class labels and array of multi-class preds.
    y_true = label_df.values.argmax(1)
    y_pred = pred_df.values.argmax(1)

    if balanced:
        # Balance the labels.
        counts = np.bincount(y_true)
        min_count = counts[counts.argmin()]

        permutation = lambda N, K: np.random.permutation(N)[:K]
        selected_ind = np.concatenate([
            np.where(y_true == label)[0][permutation(count, min_count)]
            for label, count in enumerate(counts)
        ])

        y_true = y_true[selected_ind]
        y_pred = y_pred[selected_ind]

    # Construct the confusion matrix.
    conf_df = pd.DataFrame(
        sklearn.metrics.confusion_matrix(y_true, y_pred),
        columns=label_cols, index=label_cols)

    metrics['conf_df'] = conf_df
    total = conf_df.sum().sum()
    metrics['conf_df_n'] = conf_df.astype(float) / total

    metrics['accuracy'] = np.diagonal(conf_df).sum().astype(float) / total

    results = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred)
    metrics['results_df'] = pd.DataFrame(
        np.array(results).T,
        columns=[['precision', 'recall', 'f1-score', 'support']],
        index=label_cols)

    metrics['confusion_table_fig'] = \
        vislab.dataset_viz.plot_conditional_occurrence(
            metrics['conf_df_n'], sort_by_prior=False)

    name = 'balanced dataset' if balanced else 'full dataset'
    print_metrics(metrics, name)

    return metrics


def regression_metrics(pred_df):
    # TODO
    r2_score = sklearn.metrics.r2_score(
        pred_df['label'].values, pred_df['pred'].values)
    metrics = {
        'r2_score': r2_score
    }
    return metrics


def print_metrics(metrics, name):
    print('-'*60)
    if len(name) > 0:
        name = 'on the {}'.format(name)
    print("Classification metrics {}".format(name))
    if 'results_df' in metrics:
        print(metrics['results_df'].to_string())
    if 'accuracy' in metrics:
        print("Accuracy: {}".format(metrics['accuracy']))
    if 'mcc' in metrics:
        print("Matthews' Correlation Coefficient: {}".format(metrics['mcc']))
    print('')


def get_roc_curve(y_true, y_score, title=None, with_plot=True):
    """
    Plot the [Receiver Operating Characteristic][roc] curve of the given
    true labels and confidence scores.

    [roc]: http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    auc = np.trapz(tpr, fpr)
    fig = None
    if with_plot:
        fig = vislab.results_viz.plot_curve_with_area(
            fpr, tpr, auc, 'False Positive Rate', 'True Positive Rate', 'AUC')
        ax = fig.get_axes()[0]
        ax.plot([0, 1], [0, 1], 'k--')
        if title is not None:
            ax.set_title(title)
    return fig, fpr, tpr, auc


def get_pr_curve(y_true, y_score, title=None, with_plot=True):
    """
    Plot Precision-Recall curve of the true labels and confidence scores
    and return the precision and recall vectors and the average
    precision.

    Returns
    -------
    fig: plt.Figure
    prec: ndarray
    rec: ndarray
    ap: float
    """
    prec, rec, thresh = sklearn.metrics.precision_recall_curve(y_true, y_score)
    # Make sure prec is non-increasing (prec is in reverse order)
    for i in range(len(prec) - 1):
        prec[i + 1] = max(prec[i + 1], prec[i])
    ap = np.trapz(-prec, rec)
    fig = None
    if with_plot:
        fig = vislab.results_viz.plot_curve_with_area(
            rec, prec, ap, 'Recall', 'Precision', 'AP', title)
    return fig, prec, rec, ap
