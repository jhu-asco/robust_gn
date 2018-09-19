#!/usr/bin/env python
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import least_squares
from robust_gnn.lqr_update import lqrUpdateGains
from robust_gnn.gn_residual import residual
from robust_gnn.sampling import getNominalDynamics, getNominalEllipsoids
from collections import namedtuple

Result = namedtuple('GnLqrOut', 'xs_opt, us_opt, gn_out, feedback_gains, Sigma_out')



def gn_lqr_algo(N, h, x0, u0, Sigma0, Sigmaw, integrator, cost_gains,
                obstacles, projection_matrix, ko_sqrt, xds, uds, xd_N):
    np.set_printoptions(precision=4, suppress=True)
    # Not doing matrix sqrt since diagonal
    cost_sqrt_gains = [sqrtm(gain) for gain in cost_gains]
    # Get feedback gains from LQR
    feedback_gains = np.empty((N, integrator.m, integrator.n))
    lqrUpdateGains(u0, h, N, x0, cost_gains, integrator, feedback_gains)
    print "Feedback gains: ", feedback_gains
    # Save args
    args = (h, N, x0, Sigma0, Sigmaw, xds, uds, integrator, cost_sqrt_gains,
            feedback_gains, obstacles, projection_matrix, xd_N, ko_sqrt)
    # Perform GN now
    out = least_squares(residual, u0, args=args, verbose=2)
    u_opt = out.x
    xs_opt, us_opt = getNominalDynamics(u_opt, h, N, x0, integrator)
    lqrUpdateGains(u_opt, h, N, x0, cost_gains, integrator, feedback_gains)
    Sigma_out = getNominalEllipsoids(h, xs_opt, us_opt, Sigma0, Sigmaw,
                                     projection_matrix, feedback_gains,
                                     integrator)
    return Result(xs_opt, us_opt, out, feedback_gains, Sigma_out)

    
