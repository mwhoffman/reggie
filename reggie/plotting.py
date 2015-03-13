"""
Plotting for various models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pl

__all__ = ['plot_posterior']


def _figure(fig, draw=True):
    if draw:
        fig.canvas.draw()
    return fig


def _axis(ax,
          xlabel='', ylabel='', xticks=True, yticks=True, legend=False,
          despine=True, draw=True):
    ax.axis('tight')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xticks:
        ax.set_xticklabels([])
    if not yticks:
        ax.set_yticklabels([])
    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if legend:
        ax.legend(loc=0)
    if draw:
        ax.figure.canvas.draw()
    return ax


def plot_posterior(model, xmin=None, xmax=None, data=True, predictive=False,
                   **kwargs):
    """
    Plot the marginal distribution of the given one-dimensional posterior
    model.
    """
    if model.ndata == 0 and (xmin is None or xmax is None):
        raise ValueError('model has no data and no bounds are given')

    # grab the data
    X, Y = model.data

    # get the input points
    xmin = np.min(X) if (xmin is None) else xmin
    xmax = np.max(X) if (xmax is None) else xmax
    x = np.linspace(xmin, xmax, 500)

    # get the posterior mean and confidence bands
    mu, s2 = model.get_posterior(x[:, None], predictive=predictive)
    lo = mu - 2 * np.sqrt(s2)
    hi = mu + 2 * np.sqrt(s2)

    # get the axes.
    lw = 2
    ls = '-'
    alpha = 0.25

    # get the axis
    ax = pl.gca()
    ax.cla()

    # plot the mean
    lines = ax.plot(x, mu, lw=lw, ls=ls, label='mean')
    color = lines[0].get_color()

    # plot error bars
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)
    ax.plot([], [], lw=10, color=color, alpha=alpha, label='uncertainty')

    if data:
        ax.scatter(X.ravel(), Y, zorder=5, marker='o', s=30, lw=1,
                   facecolors='none', label='data')

    # complete the axis
    _axis(ax, **kwargs)


def plot_chain(samples, names=None, **kwargs):
    # figure-level kwargs
    draw = kwargs.pop('draw', True)
    xticks = kwargs.pop('xticks', True)

    # ignored kwargs
    _ = kwargs.pop('legend', None)
    _ = kwargs.pop('yticks', True)

    samples = np.array(samples, copy=False, ndmin=2)
    samples = samples - np.min(samples, axis=0)
    samples /= np.max(samples, axis=0)

    d = samples.shape[1]
    names = ['' for _ in xrange(d)] if (names is None) else names

    fig = pl.gcf()
    fig.clf()

    for i, name in enumerate(names):
        ax = fig.add_subplot(d, 1, i+1)
        ax.plot(samples[:, i])
        _axis(ax,
              xticks=xticks and (i == d-1),
              yticks=False,
              ylabel=name,
              draw=False,
              **kwargs)

    # complete the figure
    _figure(fig, draw)


def plot_pairs(samples, names=None, **kwargs):
    # figure-level kwargs
    draw = kwargs.pop('draw', True)
    ticks = kwargs.pop('ticks', True)

    # ignored kwargs
    _ = kwargs.pop('legend', None)
    _ = kwargs.pop('xticks', None)
    _ = kwargs.pop('yticks', None)

    d = samples.shape[1]
    names = ['' for _ in xrange(d)] if (names is None) else names

    fig = pl.gcf()
    fig.clf()

    for i in xrange(d):
        for j in xrange(i+1, d):
            ax = fig.add_subplot(d-1, d-1, (j-1)*(d-1)+i+1)
            ax.scatter(samples[:, i], samples[:, j], edgecolor='white',
                       alpha=0.1)
            _axis(ax,
                  xticks=ticks and (j == d-1),
                  yticks=ticks and (i == 0),
                  xlabel=names[i] if (j == d-1) else '',
                  ylabel=names[j] if (i == 0) else '',
                  draw=False,
                  **kwargs)

    # complete the figure
    _figure(fig, draw)
