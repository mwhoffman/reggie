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
        # update the temporary model using parameters in the transformed space
        model_.set_params(theta, True)

        # get the log-probability and its gradient in the untransformed space
        logp0, dlogp0 = model_.get_logprior(True)
        logp1, dlogp1 = model_.get_loglike(True)

        # form the posterior probability and multiply by the grad factor which
        # gives us the gradient in the transformed space (via the chain rule)
        logp = -(logp0 + logp1)
        dlogp = -(dlogp0 + dlogp1) * model_.get_gradfactor()

        return logp, dlogp

    # optimize the model
    theta, _, _ = so.fmin_l_bfgs_b(func=objective,
                                   x0=model.get_params(True),
                                   bounds=model.get_support(True))

    # make sure that the model is using the correct hypers
    model.set_params(theta, True)
