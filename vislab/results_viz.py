import matplotlib.pyplot as plt


def plot_binary_metrics(df):
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    df[['ap', 'mcc']].plot(
        ax=ax, kind='bar')
    ax.set_ylim([0, 1])
    fig.autofmt_xdate()
    return fig


def plot_top_k_accuracies(accuracies, top_k):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(1, top_k + 1), accuracies[:top_k], 's--')
    ax.set_xlim([1, top_k])
    ax.set_xticks(range(1, top_k + 1))
    ax.set_xlabel('K')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Top-K Accuracy')
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
