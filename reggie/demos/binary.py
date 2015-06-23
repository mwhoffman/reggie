import os
import numpy as np
import mwhutils.plotting as mp
import mwhutils.grid as mg

import reggie as rg


if __name__ == '__main__':
    cdir = os.path.abspath(os.path.dirname(__file__))
    data = np.load(os.path.join(cdir, 'xy.npz'))

    # create the GP and optimize the model
    gp1 = rg.make_gp(0.1, 1.0, 0.1)
    gp1.add_data(data['X'], data['y'])
    gp1.optimize()

    xmin = data['X'].min()
    xmax = data['X'].max()

    like = rg.likelihoods.Probit()
    kern = gp1._post.kern.copy()
    mean = gp1._post.mean.copy()

    f = gp1.sample_f(100)
    X = mg.uniform([(xmin, xmax)], 1000)
    Y = like.sample(f.get(X))

    gp2 = rg.GP(like, kern, mean, inference='laplace')
    gp2.add_data(X, Y)
    gp2.optimize()

    # create the figure
    fig = mp.figure(1, 2)
    fig.hold()

    # get the posterior moments for the first model
    n = 500
    x = np.linspace(xmin, xmax, n)
    mu, s2 = gp1.predict(x[:, None])

    fig[0].plot(x, mu, 2*np.sqrt(s2))
    fig[0].scatter(data['X'], data['y'])
    fig[0].xlabel = 'inputs, X'
    fig[0].ylabel = 'outputs, Y'
    fig[0].title = 'Basic GP'

    # get the posterior moments for the second model
    mu, s2 = gp2.predict(x[:, None])
    fig[1].plot(x, mu, 2*np.sqrt(s2), label='posterior mean')
    fig[1].plot(x, f.get(x[:, None]), label='sample function')
    fig[1].scatter(gp2.data[0], gp2.data[1], label='observed data')
    fig[1].xlabel = 'inputs, X'
    fig[1].title = 'Binary GP\n(with sampled function)'

    # show the figure
    fig.draw()
    mp.show()
