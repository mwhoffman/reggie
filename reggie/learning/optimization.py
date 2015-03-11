"""
Perform type-II maximum likelihood to fit the hyperparameters of a GP model.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.optimize as so

__all__ = ['optimize']


def optimize(model, raw=False):
    """
    Perform type-II maximum likelihood to fit GP hyperparameters.
    """
    # use a copy of the model so we don't modify it while we're optimizing
    model = model.copy()

    # define the objective to MINIMIZE
    def objective(theta):
        # update the temporary model using parameters in the transformed space
        model.set_params(theta, True)

        # get the log-probability and its gradient in the untransformed space
        logp0, dlogp0 = model.get_logprior(True)
        logp1, dlogp1 = model.get_loglike(True)

        # form the posterior probability and multiply by the grad factor which
        # gives us the gradient in the transformed space (via the chain rule)
        logp = -(logp0 + logp1)
        dlogp = -(dlogp0 + dlogp1) * model.get_gradfactor()

        return logp, dlogp

    # get rid of any infinite bounds.
    bounds = model.get_bounds(True)
    isinf = np.isinf(bounds)
    bounds = np.array(bounds, dtype=object)
    bounds[isinf] = None
    bounds = map(tuple, bounds)

    # optimize the model
    theta, _, _ = so.fmin_l_bfgs_b(func=objective,
                                   x0=model.get_params(True),
                                   bounds=bounds)

    if raw:
        # return the parameters in the transformed space
        model = theta
    else:
        model.set_params(theta, True)

    return model
