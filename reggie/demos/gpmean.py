"""
Demo showing how to use a posterior GP mean as the prior mean for
a subsequent GP.
"""

import numpy as np
import reggie

from ezplot import figure, show
from reggie import make_gp


def main():
    """Run the demo."""
    # generate random data from a gp prior
    rng = np.random.RandomState(0)
    gp = make_gp(0.1, 1.0, 1.0, kernel='matern3')
    X = rng.uniform(-2, 2, size=(20, 1))
    Y = gp.sample(X, latent=False, rng=rng)

    # create a new GP and optimize its hyperparameters
    gp = make_gp(1, 1, 1, kernel='se')
    gp.add_data(X, Y)
    gp.optimize()

    # get the posterior moments
    x = np.linspace(-2, 2, 500)
    mu, s2 = gp.predict(x[:, None])

    # plot the posterior of gp
    fig = figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot_banded(x, mu, 2*np.sqrt(s2), label='posterior mean')
    ax.scatter(X.ravel(), Y, label='observed data')
    ax.legend(loc=0, scatterpoints=1)
    ax.set_title('Posterior GP')
    ax.set_xlabel('inputs, X')
    ax.set_ylabel('outputs, Y')
    ax.figure.canvas.draw()

    # initialize new GP with previous GP's posterior mean as prior mean
    like = gp._like
    kern = gp._kern
    mean = reggie.means.GPMean(gp)
    gp_warm = reggie.GP(like, kern, mean)

    # prior prediction
    mu, s2 = gp_warm.predict(x[:, None])

    # plot the prior of gp_warm
    ax = fig.add_subplot(1, 2, 2, sharey=ax)
    ax.plot_banded(x, mu, 2*np.sqrt(s2), color='k', label='warm prior mean')

    # generate new data
    X = rng.uniform(-2, 2, size=(5, 1))
    Y = gp_warm.sample(X, latent=False, rng=rng)

    # fit and predict
    gp_warm.add_data(X, Y)
    mu, s2 = gp_warm.predict(x[:, None])

    # plot the posterior and new data
    ax.plot_banded(x, mu, 2*np.sqrt(s2), color='g', label='new posterior mean')
    ax.scatter(X.ravel(), Y, marker='s', label='new observed data')
    ax.legend(loc=0, scatterpoints=1)
    ax.set_title('Warm prior GP')
    ax.set_xlabel('inputs, X')
    ax.set_ylabel('outputs, Y')
    ax.figure.canvas.draw()

    # show it
    show()


if __name__ == '__main__':
    main()
