import os
import numpy as np
import matplotlib.pyplot as pl

import reggie.gps
import reggie.plotting


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    gp = reggie.gps.BasicGP(0.1, 1.0, 0.1)
    gp.add_data(data['X'], data['y'])
    gp.optimize()

    pl.figure(1)
    reggie.plotting.plot_posterior(gp)
