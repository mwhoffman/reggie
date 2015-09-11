"""
Demo showing the approximation of 1d kernels with random Fourier features.
"""

import numpy as np

from ezplot import figure, show
from reggie.kernels import SE, Matern


def main():
    """Run the demo."""
    # grab some kernels
    kernels = [SE(1, 1),
               Matern(1, 1, d=5)]

    # the points we'll test at
    x = np.linspace(-5, 5, 500)
    n = 1000

    # seed the rng and create a figure
    rng = np.random.RandomState(0)
    fig = figure()

    for i, kernel in enumerate(kernels):
        # the true kernel
        k1 = kernel.get_kernel([[0]], x[:, None]).ravel()

        # the approximation
        W, a = kernel.sample_spectrum(n, rng)
        b = rng.rand(n) * 2 * np.pi
        k2 = np.dot(np.cos(np.dot(x[:, None], W.T) + b), np.cos(b)) * 2 * a / n

        # plot it
        ax = fig.add_subplot(len(kernels), 1, 1+i)
        ax.plot(x, k1, label='true kernel')
        ax.plot(x, k2, label='feature approximation')
        ax.set_title(kernel.__class__.__name__)

        if i == 0:
            ax.legend(loc=0)

    # draw everything
    fig.canvas.draw()
    show()


if __name__ == '__main__':
    main()
