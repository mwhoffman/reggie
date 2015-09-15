"""
Demo showing GP predictions in 1d and optimization of the hyperparameters.
"""

import numpy as np

from ezplot import figure, show
from reggie import make_gp


def main():
    """Run the demo."""
    # generate random data from a gp prior
    rng = np.random.RandomState(0)
    gp = make_gp(0.1, 1.0, 0.1, kernel='matern1')
    X = rng.uniform(-2, 2, size=(20, 1))
    Y = gp.sample(X, latent=False, rng=rng)

    # create a new GP and optimize its hyperparameters
    gp = make_gp(1, 1, 1, kernel='se')
    gp.add_data(X, Y)
    gp.optimize()

    # get the posterior moments
    x = np.linspace(X.min(), X.max(), 500)
    mu, s2 = gp.predict(x[:, None])

    # plot the posterior
    ax = figure().gca()
    ax.plot_banded(x, mu, 2*np.sqrt(s2), label='posterior mean')
    ax.scatter(X.ravel(), Y, label='observed data')
    ax.legend(loc=0)
    ax.set_title('Basic GP')
    ax.set_xlabel('inputs, X')
    ax.set_ylabel('outputs, Y')

    # draw/show it
    ax.figure.canvas.draw()
    show()


if __name__ == '__main__':
    main()
