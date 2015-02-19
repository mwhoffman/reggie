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


def plot_posterior(model, xmin=None, xmax=None):
    if not isinstance(model, PosteriorModel):
        raise ValueError('model must be a PosteriorModel instance')

    if model.ndata == 0 and (xmin is None or xmax is None):
        raise ValueError('model has no data and no bounds are given')

    # grab the data
    X, Y = model.data

    # get the input points
    xmin = X[:, 0].min() if (xmin is None) else xmin
    xmax = X[:, 0].max() if (xmax is None) else xmax
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
    color = next(ax._get_lines.color_cycle)
    alpha = 0.25

    ax.fill_between(x, lo, hi, color=color, alpha=alpha)
    ax.plot(x, mu, lw=lw, ls=ls, color=color, label='mean')
    ax.plot([], [], lw=10, color=color, alpha=alpha, label='uncertainty')
    ax.scatter(X.ravel(), Y,
               marker='o', s=30, lw=1, facecolors='none', label='data')

    ax.axis('tight')
