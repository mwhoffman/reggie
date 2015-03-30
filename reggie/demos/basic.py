import os
import numpy as np
import mwhutils.plotting as mp

import reggie as rg


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    X = data['X']
    Y = data['y']

    gp = rg.BasicGP(0.1, 1.0, 0.1)
    gp.add_data(X, Y)
    gp.optimize()

    x = np.linspace(X.min(), X.max(), 500)
    mu, s2 = gp.get_posterior(x[:, None])

    fig = mp.figure(1)
    fig.plot_banded(x, mu, 2*np.sqrt(s2))
    fig.scatter(X.ravel(), Y)
    fig.draw()
