import numpy as np
import mwhutils.plotting as mp
import mwhutils.random as random
import reggie as rg


if __name__ == '__main__':
    kernels = [
        rg.kernels.SE(1, 1),
        rg.kernels.Matern(1, 1, d=5)]

    # seed the rng
    rng = random.rstate()
    x = np.linspace(-5, 5, 500)
    n = 1000

    fig = mp.figure(len(kernels))
    fig.hold()

    # the points we'll test at
    for i, kernel in enumerate(kernels):
        # the true kernel
        k1 = kernel.get_kernel([[0]], x[:, None]).ravel()

        # the approximation
        W, a = kernel.sample_spectrum(n, rng)
        b = rng.rand(n) * 2 * np.pi
        k2 = np.dot(np.cos(np.dot(x[:, None], W.T) + b), np.cos(b)) * 2 * a / n

        # plot it
        fig[i].plot(x, k1, label='true kernel')
        fig[i].plot(x, k2, label='feature approximation')
        fig[i].title = kernel.__class__.__name__
        fig[i].draw()

    mp.show()
