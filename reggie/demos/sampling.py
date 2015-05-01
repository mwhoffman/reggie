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
    gp = rg.make_gp(0.1, 1.0, 0.1)

    # set the hyperpriors
    gp.params['like.sn2'].set_prior('lognormal', 0, 10)
    gp.params['kern.rho'].set_prior('lognormal', 0, 100)
    gp.params['kern.ell'].set_prior('lognormal', 0, 10)
    gp.params['mean.bias'].set_prior('normal', 0, 20)

    # put each hyperparameter into a separate block
    gp.params.block = range(gp.params.size)

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
    fig = mp.figure(1)
    fig.hold()
    fig.plot_banded(x, mu, 2*np.sqrt(s2))
    fig.scatter(X.ravel(), Y)
    fig.xlabel = 'inputs, X'
    fig.ylabel = 'outputs, Y'
    fig.draw()

    # plot the samples
    mp.plot_pairs(mcmc.get_samples(), fig=2)

    # block if we're in non-interactive mode
    mp.show()
