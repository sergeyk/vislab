import matplotlib.pyplot as plt


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
