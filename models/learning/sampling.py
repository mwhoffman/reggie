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
    theta0 = model.get_params(True)

    # sample a random direction
    direction = rng.randn(theta0.shape[0])
    direction /= np.sqrt(np.sum(direction**2))

    # this computes the new model (ie statistics) and its probability
    def get_logprob(z):
        model_ = model.copy(theta0 + direction*z, True)
        logp0, _ = model_.get_logprior()
        logp1, _ = model_.get_loglike()
        return logp0 + logp1, model_

    upper = sigma*rng.rand()
    lower = upper - sigma

    logprob0 = np.log(rng.rand())
    logprob0 += model.get_logprior()[0]
    logprob0 += model.get_loglike()[0]

    if step_out:
        for _ in xrange(max_steps_out):
            if get_logprob(lower)[0] <= logprob0:
                break
            lower -= sigma
        for _ in xrange(max_steps_out):
            if get_logprob(upper)[0] <= logprob0:
                break
            upper += sigma

    while True:
        z = (upper - lower)*rng.rand() + lower
        logprob, model_ = get_logprob(z)
        if np.isnan(logprob):
            raise Exception("Slice sampler got a NaN")
        if logprob > logprob0:
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
