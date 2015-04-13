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

    # evaluate the posterior at some test points
    x = np.linspace(X.min(), X.max(), 500)
    mu, s2 = gp.predict(x[:, None])

    # plot the posterior
    fig = mp.figure()
    fig.hold()
    fig.plot_banded(x, mu, 2*np.sqrt(s2))
    fig.scatter(X.ravel(), Y)
    fig.set_xlabel('inputs, X')
    fig.set_ylabel('outputs, Y')
    fig.draw()

    # plot the samples
    mp.plot_pairs(mcmc.get_samples(), mcmc.names)

    # block if we're in non-interactive mode
    mp.show()
