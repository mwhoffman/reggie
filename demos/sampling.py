import os
import numpy as np
import matplotlib.pyplot as pl

import models.gps
import models.learning
import models.plotting


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    # create a model
    gp = models.gps.BasicGP(0.1, 1.0, 0.1)

    # set the hyperpriors
    gp.set_prior('sn2', 'uniform', 0.001, 1)
    gp.set_prior('rho', 'uniform', 0.001, 10)
    gp.set_prior('ell', 'uniform', 0.001, 1)
    gp.set_prior('mean', 'uniform', -3, 3)

    # add data and optimize
    gp.add_data(data['X'], data['y'])
    gp.optimize()

    # sample hyperparameters
    mcmc = models.learning.MetaMCMC(gp, n=1000, rng=0)

    # optimize and plot
    pl.figure(1)
    pl.subplot(121)
    models.plotting.plot_posterior(gp, draw=False)
    pl.axis(ymax=3, ymin=-2)
    pl.title('MAP')

    pl.subplot(122)
    models.plotting.plot_posterior(mcmc, draw=False)
    pl.axis(ymax=3, ymin=-2)
    pl.title('MCMC')
    pl.draw()
