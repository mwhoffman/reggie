import os
import numpy as np

from ezplot import figure, show
from reggie import make_gp


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    X = data['X']
    Y = data['y']
    U = np.linspace(X.min(), X.max(), 10)[:, None]

    # create a basic GP and switch to sparse inference
    gp = make_gp(0.1, 1.0, 0.1, inf='fitc', U=U)
    gp.add_data(X, Y)
    gp.optimize()

    # get the posterior moments
    x = np.linspace(X.min(), X.max(), 500)
    mu, s2 = gp.predict(x[:, None])

    # plot the posterior
    ax = figure().gca()
    ax.plot_banded(x, mu, 2*np.sqrt(s2), label='posterior mean')
    ax.scatter(X, Y, label='observed data')
    ax.scatter(U, np.full_like(U, -1), marker='x', label='inducing points')
    ax.legend(loc=0)
    ax.set_xlabel('inputs, X')
    ax.set_ylabel('outputs, Y')
    ax.set_title('Sparse GP (FITC)')

    # show the figure
    ax.figure.canvas.draw()
    show()
