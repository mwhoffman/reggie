"""
Perform parameter sampling.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from mwhutils.random import rstate

__all__ = ['sample']


def slice_sample(model,
                 sigma=1.0,
                 step_out=True,
                 max_steps_out=1000,
                 rng=None):
    """
    Implementation of a generic slice sampling step which takes a model
    instance and returns a new model instance.
    """
    rng = rstate(rng)

    # get the initial parameter assignments
    theta0 = model.get_params()
    support = model.get_support()

    # sample a random direction
    direction = rng.randn(theta0.shape[0])
    direction /= np.sqrt(np.sum(direction**2))

    # this updates the new model (ie computing its sufficient statistics) and
    # returns the posterior probability and the model itself.
    def get_logp(z):
        theta = theta0 + direction*z
        if np.any(theta < support[:, 0]) or np.any(theta > support[:, 1]):
            model_ = None
            logp = -np.inf
        else:
            model_ = model.copy(theta)
            logp = model_.get_logprior() + model_.get_loglike()
        return model_, logp

    upper = sigma*rng.rand()
    lower = upper - sigma
    logp0 = np.log(rng.rand()) + model.get_logprior() + model.get_loglike()

    if step_out:
        for _ in xrange(max_steps_out):
            if get_logp(lower)[1] <= logp0:
                break
            lower -= sigma
        for _ in xrange(max_steps_out):
            if get_logp(upper)[1] <= logp0:
                break
            upper += sigma

    while True:
        z = (upper - lower)*rng.rand() + lower
        model_, logp = get_logp(z)
        if logp > logp0:
            break
        elif z < 0:
            lower = z
        elif z > 0:
            upper = z
        else:
            raise Exception("Slice sampler shrank to zero!")

    return model_


def sample(model, n, raw=False, rng=None):
    rng = rstate(rng)
    models = []
    models.append(slice_sample(model))
    for _ in xrange(n-1):
        models.append(slice_sample(models[-1]))
    if raw:
        models = np.array([_.get_params() for _ in models])
    return models
