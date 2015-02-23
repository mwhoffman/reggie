"""
Perform type-II maximum likelihood to fit the hyperparameters of a GP model.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import scipy.optimize as so

__all__ = ['optimize']


def optimize(gp):
    """
    Perform type-II maximum likelihood to fit GP hyperparameters.
    """
    # use a copy of the gp so we don't modify it while we're optimizing
    gp_ = gp.copy()

    # define the objective to MINIMIZE
    def objective(theta):
        gp_.set_params(theta, True)
        logp0, dlogp0 = gp_.get_logprior()
        logp1, dlogp1 = gp_.get_loglike()

        logp = -(logp0 + logp1)
        dlogp = -gp_.transform_grad(theta, dlogp0 + dlogp1)

        return logp, dlogp

    # optimize the model
    theta, _, _ = so.fmin_l_bfgs_b(objective, gp.get_params(True))

    # make sure that the gp is using the correct hypers
    gp.set_params(theta, True)
