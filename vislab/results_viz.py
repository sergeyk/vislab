import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np


def plot_roc_curve(y_true, y_score):
    """
    Plot the [Receiver Operating Characteristic][roc] curve of the given
    true labels and confidence scores.

    [roc]: http://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = np.trapz(tpr, fpr)
    fig = plot_curve_with_area(
        fpr, tpr, auc, 'False Positive Rate', 'True Positive Rate', 'AUC')
    ax = fig.get_axes()[0]
    ax.plot([0, 1], [0, 1], 'k--')
    return fig, fpr, tpr, auc


def plot_pr_curve(y_true, y_score):
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
    prec, rec, thresholds = metrics.precision_recall_curve(y_true, y_score)
    # Make sure prec is non-increasing (prec is in reverse order)
    for i in range(len(prec) - 1):
        prec[i + 1] = max(prec[i + 1], prec[i])
    ap = np.trapz(-prec, rec)
    fig = plot_curve_with_area(
        rec, prec, ap, 'Recall', 'Precision', 'AP')
    return fig, prec, rec, ap


def plot_curve_with_area(x, y, area, xlabel, ylabel, area_label):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'k-')
    ax.bar(0, area, 1, alpha=0.2)
    ax.text(.05, area - 0.05, '{}: {:.3f}'.format(area_label, area))
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig
