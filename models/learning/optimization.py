"""
Perform type-II maximum likelihood to fit the hyperparameters of a GP model.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import scipy.optimize as so

__all__ = ['optimize']


def optimize(model):
    """
    Perform type-II maximum likelihood to fit GP hyperparameters.
    """
    # use a copy of the model so we don't modify it while we're optimizing
    model_ = model.copy()

    # define the objective to MINIMIZE
    def objective(theta):
        model_.set_params(theta, True)
        logp0, dlogp0 = model_.get_logprior()
        logp1, dlogp1 = model_.get_loglike()

        logp = -(logp0 + logp1)
        dlogp = -model_.transform_grad(theta, dlogp0 + dlogp1)

        return logp, dlogp

    # optimize the model
    theta, _, _ = so.fmin_l_bfgs_b(objective, model.get_params(True))

    # make sure that the model is using the correct hypers
    model.set_params(theta, True)
