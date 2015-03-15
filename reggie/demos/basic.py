import os
import numpy as np
import matplotlib.pyplot as pl

import reggie as rg
import reggie.plotting


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    gp = rg.BasicGP(0.1, 1.0, 0.1)
    gp.add_data(data['X'], data['y'])
    gp.optimize()

    pl.figure(1)
    rg.plotting.plot_posterior(gp)
