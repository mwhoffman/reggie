"""
Plotting for various models.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as pl

from .core.models import PosteriorModel

__all__ = ['plot_posterior']


def plot_posterior(model, xmin=None, xmax=None, data=True, draw=True,
                   despine=False):
    """
    Plot the marginal distribution of the given one-dimensional posterior
    model.
    """

    if not isinstance(model, PosteriorModel):
        raise ValueError('model must be a PosteriorModel instance')

    if model.ndata == 0 and (xmin is None or xmax is None):
        raise ValueError('model has no data and no bounds are given')

    # grab the data
    X, Y = model.data

    # get the input points
    xmin = np.min(X) if (xmin is None) else xmin
    xmax = np.max(X) if (xmax is None) else xmax
    x = np.linspace(xmin, xmax, 500)

    # get the posterior mean and confidence bands
    mu, s2 = model.get_posterior(x[:, None])
    lo = mu - 2 * np.sqrt(s2)
    hi = mu + 2 * np.sqrt(s2)

    # get the axes.
    ax = pl.gca()
    ax.cla()

    lw = 2
    ls = '-'
    alpha = 0.25

    # plot the mean
    lines = ax.plot(x, mu, lw=lw, ls=ls, label='mean')
    color = lines[0].get_color()

    # plot error bars
    ax.fill_between(x, lo, hi, color=color, alpha=alpha)
    ax.plot([], [], lw=10, color=color, alpha=alpha, label='uncertainty')

    if data:
        ax.scatter(X.ravel(), Y, zorder=5, marker='o', s=30, lw=1,
                   facecolors='none', label='data')

    ax.axis('tight')

    if despine:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if draw:
        ax.figure.canvas.draw()
