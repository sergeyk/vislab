"""
All credit for rstyle, rhist, rbox goes to [messymind.net][1].
(With some additions from the comments section.)

Additional credit (husl_gen, rbar) goes to [Rob Story][2].

[1]: http://messymind.net/2012/07/making-matplotlib-look-like-ggplot/
[2]: http://nbviewer.ipython.org/urls/raw.github.com/\
wrobstory/climatic/master/examples/ggplot_styling_for_matplotlib.ipynb
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import husl
import pylab
try:
    import mpltools.style
    mpltools.style.use('ggplot')
    # Colors from http://mbostock.github.io/protovis/docs/color.html
    matplotlib.rcParams['axes.color_cycle'] = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
except:
    pass


def husl_gen():
    """
    Generate random set of HUSL colors, one dark, one light.
    """
    hue = np.random.randint(0, 360)
    saturation, lightness = np.random.randint(0, 100, 2)
    husl_dark = husl.husl_to_hex(hue, saturation, lightness / 3)
    husl_light = husl.husl_to_hex(hue, saturation, lightness)
    return str(husl_dark), str(husl_light)


def rstyle(ax, xlog=False, ylog=False):
    """
    Styles x,y axes to appear like ggplot2.

    Must be called after all plot and axis manipulation operations have been
    carried out, as it needs to know the final tick spacing.
    """
    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('#f3f3f3')
    ax.set_axisbelow(True)

    # #Set minor tick spacing to 1/2 of the major ticks
    # if not xlog:
    #     ax.xaxis.set_minor_locator((pylab.MultipleLocator((
    #         plt.xticks()[0][1] - plt.xticks()[0][0]) / 2.0)))
    # if not ylog:
    #     ax.yaxis.set_minor_locator((pylab.MultipleLocator((
    #         plt.yticks()[0][1] - plt.yticks()[0][0]) / 2.0)))

    #Remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)

    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("lightgray")
        line.set_markeredgewidth(1.4)

    #Remove the minor tick lines
    for line in (ax.xaxis.get_ticklines(minor=True) +
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)

    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.legend()
    legend = ax.get_legend()
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor('#f3f3f3')


def rbar(ax, left, height, **kwargs):
    """
    Create a bar plot with default style parameters to look like ggplot2.

    kwargs can be passed to changed other parameters
    """
    defaults = {'facecolor': '0.15',
                'edgecolor': '0.28',
                'linewidth': 1,
                'width': 1}

    for x, y in defaults.iteritems():
        kwargs.setdefault(x, y)

    return ax.bar(left, height, **kwargs)


def rfill(ax, x_range, dist, **kwargs):
    """
    Create a density plot to resemble ggplot2.

    kwargs can be passed to change other parameters.
    """
    defaults = {'linewidth': 2.0,
                'alpha': 0.4}

    for x, y in defaults.iteritems():
        kwargs.setdefault(x, y)

    # Make edge color a darker shade of facecolor.
    patches = ax.fill(x_range, dist, **kwargs)
    for patch in patches:
        fc = patch.get_facecolor()
        patch.set_edgecolor(tuple(x * 0.5 for x in fc[:3]) + (fc[3],))
    return ax


def rhist(ax, data, **kwargs):
    """
    Create a hist plot with default style parameters to look like ggplot2.

    kwargs can be passed to changed other parameters.
    """
    defaults = {'facecolor': '0.3',
                'edgecolor': '0.36',
                'linewidth': 1,
                'rwidth': 1}

    for x, y in defaults.iteritems():
        kwargs.setdefault(x, y)

    return ax.hist(data, **kwargs)


def rbox(ax, data, **keywords):
    """
    Create a ggplot2 style boxplot, is eqivalent to calling ax.boxplot
    with the following additions:

    Keyword arguments:
    colors -- array-like collection of colours for box fills
    names -- array-like collection of box names which are passed on as
    tick labels
    """
    hasColors = 'colors' in keywords
    if hasColors:
        colors = keywords['colors']
        keywords.pop('colors')

    if 'names' in keywords:
        ax.tickNames = plt.setp(ax, xticklabels=keywords['names'])
        keywords.pop('names')

    bp = ax.boxplot(data, **keywords)
    pylab.setp(bp['boxes'], color='black')
    pylab.setp(bp['whiskers'], color='black', linestyle='solid')
    pylab.setp(bp['fliers'], color='black', alpha=.9, marker='o', markersize=3)
    pylab.setp(bp['medians'], color='black')

    numBoxes = len(data)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX, boxY)

        if hasColors:
            boxPolygon = pylab.Polygon(
                boxCoords, facecolor=colors[i % len(colors)])
        else:
            boxPolygon = pylab.Polygon(boxCoords, facecolor='0.95')

        ax.add_patch(boxPolygon)
    return bp
