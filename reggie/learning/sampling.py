"""
Perform parameter sampling.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mwhutils.random import rstate

__all__ = ['sample']


def slice_sample(model, sigma=1.0, max_steps=1000, rng=None):
    """
    Implementation of a generic slice sampling step which takes a model
    instance and returns a new model instance.
    """
    rng = rstate(rng)

    # the initial parameter values
    theta0 = model.params.get_value(transform=True)

    # posterior including the log-determinant of the transform's jacobian
    logp0 = model.params.get_logprior() + model.get_loglike()
    logp0 += np.sum(np.log(model.params.gradfactor))

    def get_logp(theta):
        try:
            # get the model at the new parameters
            model_ = model.copy(theta, transform=True)

            # compute the posterior probability and again we need to include
            # the log-determinant of the transform's jacobian
            logp = model_.params.get_logprior() + model_.get_loglike()
            logp += np.sum(np.log(model_.params.gradfactor))

        except ValueError:
            # we tried to set parameters that don't lie in the support of
            # either the variable itself or its prior.
            model_ = None
            logp = -np.inf

        return model_, logp

    for block in model.params.blocks:
        # sample a random direction
        direction = np.zeros_like(theta0)
        direction[block] = rng.randn(len(block))
        direction /= np.sqrt(np.sum(direction**2))

        upper = sigma*rng.rand()
        lower = upper - sigma
        alpha = np.log(rng.rand())

        for _ in xrange(max_steps):
            if get_logp(theta0 + direction*lower)[1] <= logp0 + alpha:
                break
            lower -= sigma

        for _ in xrange(max_steps):
            if get_logp(theta0 + direction*upper)[1] <= logp0 + alpha:
                break
            upper += sigma

        while True:
            z = (upper - lower)*rng.rand() + lower
            theta = theta0 + direction*z
            model_, logp = get_logp(theta)
            if logp > logp0 + alpha:
                break
            elif z < 0:
                lower = z
            elif z > 0:
                upper = z
            else:
                raise RuntimeError("Slice sampler shrank to zero!")

        # make sure to update our starting point
        theta0 = theta
        logp0 = logp

    return model_


def sample(model, n, raw=False, rng=None):
    rng = rstate(rng)
    models = []
    for _ in xrange(n):
        model = slice_sample(model, rng=rng)
        models.append(model)
    if raw:
        models = np.array([m.params.get_value() for m in models])
    return models
