import os
import numpy as np
import mwhutils.plotting as mp

import reggie as rg


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    X = data['X']
    Y = data['y']

    # create a model
    gp = rg.BasicGP(0.1, 1.0, 0.1)

    # set the hyperpriors
    gp.set_prior('sn2', 'lognormal', 0, 10)
    gp.set_prior('rho', 'lognormal', 0, 100)
    gp.set_prior('ell', 'lognormal', 0, 10)
    gp.set_prior('mean', 'normal', 0, 20)

    gp.set_block('rho', 1)
    gp.set_block('mean', 2)
    gp.set_block('ell', 3)

    # add data and optimize
    gp.add_data(data['X'], data['y'])
    gp.optimize()

    # sample hyperparameters
    mcmc = rg.MetaMCMC(gp, n=1000, rng=None)
    samples = mcmc.get_samples()
    names = mcmc.get_names()

    x = np.linspace(X.min(), X.max(), 500)
    mu, s2 = gp.get_posterior(x[:, None])
    er = 2*np.sqrt(s2)

    # plot
    fig = mp.figure(1)
    fig.plot_banded(x, mu, mu-er, mu+er)
    fig.scatter(X.ravel(), Y)
    fig.draw()
