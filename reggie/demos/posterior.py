"""
Demo showing the typical behavior of posterior updating with a GP prior.
"""

import numpy as np

from ezplot import figure, show
from reggie import make_gp


def main():
    """Run the demo."""
    # generate random data from a gp prior
    rng = np.random.RandomState(1)
    N = 5
    X = rng.uniform(-2, 2, size=(N, 1))
    Y = rng.uniform(-2, 2, size=N)
    x = np.linspace(X.min(), X.max(), 500)

    # create a GP and sample its prior
    gp = make_gp(0.01, 1, 0.3, kernel='se')
    pr_fs = gp.sample(x[:, None], 3, rng=rng)
    pr_mu, pr_s2 = gp.predict(x[:, None])

    # add data and sample the posterior
    gp.add_data(X, Y)
    po_fs = gp.sample(x[:, None], 3, rng=rng)
    po_mu, po_s2 = gp.predict(x[:, None])

    # plot the posterior
    fig = figure(w_pad=3)
    ax1 = fig.add_subplotspec((1, 2), (0, 0), hidexy=True)
    ax2 = fig.add_subplotspec((1, 2), (0, 1), hidexy=True, sharey=ax1)

    ax1.plot_banded(x, pr_mu, 3*np.sqrt(pr_s2))
    ax1.plot(x, pr_fs.T, ls='--')
    ax1.set_title('prior')
    ax1.set_ylim(-3.5, 3.5)

    ax2.plot_banded(x, po_mu, 3*np.sqrt(po_s2))
    ax2.plot(x, po_fs.T, ls='--')
    ax2.scatter(X.ravel(), Y, s=80, marker='+', zorder=3, lw=3)
    ax2.set_title('posterior')

    # draw/show it
    fig.canvas.draw()
    show()


if __name__ == '__main__':
    main()
