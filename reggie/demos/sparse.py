import os
import numpy as np
import mwhutils.plotting as mp

import reggie as rg


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    X = data['X']
    Y = data['y']
    U = np.linspace(X.min(), X.max(), 10)[:, None]

    # create the GP and optimize the model
    gp = rg.BasicGP(0.1, 1.0, 0.1)
    gp.add_data(X, Y)
    gp.optimize()

    gp = gp.switch_inference(rg.models.gpinference.FITC(U))
