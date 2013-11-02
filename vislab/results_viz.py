import numpy as np
import matplotlib.pyplot as plt
import vislab.gg


def plot_df_bar(df, columns=None, figsize=(16, 4), fontsize=13):
    """
    Used to plot AP vs MCC for a single feature, or AP between features.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if columns is not None:
        df = df[columns]
    df.plot(ax=ax, kind='bar')
    ax.set_ylim([0, 1])
    ax.set_yticks(np.arange(11) / 10.)
    fig.autofmt_xdate()

    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.2),
        ncol=3, fancybox=True, shadow=True, prop={'size': fontsize})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    vislab.gg.rstyle(ax)

    return fig


def plot_top_k_accuracies(accuracies_df, top_k=5, font_size=13):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracies_df.ix[range(top_k + 1)].plot(ax=ax, style='s--')
    ax.set_xlim([1, top_k])
    ax.set_xticks(range(1, top_k + 1))
    ax.set_xlabel('K')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Top-K Accuracy')

    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, 1.35),
        ncol=2, fancybox=True, shadow=True, prop={'size': font_size})

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(font_size)

    vislab.gg.rstyle(ax)

    return fig


def plot_curve_with_area(x, y, area, xlabel, ylabel, area_label, title=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'k-')
    ax.bar(0, area, 1, alpha=0.2)
    ax.text(.05, area - 0.05, '{}: {:.3f}'.format(area_label, area))
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return fig
