#!/usr/bin/env python
import numpy as np
from linearized_ellipsoid_propagation import propagateEllipsoid
from obstacle_residual import obstacleResidual

def residual(u_in, h, N, x0, Sigma0, Sigmaw, xds, uds, integrator, cost_gains, feedback_gains, obstacles, projection_matrix, xd_N=None):
    us = u_in.reshape(N, -1)
    w = np.zeros_like(x0)
    x_current = x0
    Sigma_current = Sigma0
    residual_list = []
    Qsqrt, Rsqrt, Qfsqrt = cost_gains
    for i, u in enumerate(us):
        # Create residual
        if xds is not None:
            x_diff = x_current - xds[i]
            residual_list.append(np.dot(Qsqrt,x_diff))
        if uds is not None:
            u_diff = u - uds[i]
            residual_list.append(np.dot(Rsqrt, u_diff))

        for  obstacle in obstacles:
            residual_list.append(obstacleResidual(Sigma_current, x_current, obstacle, projection_matrix))
        x_current = integrator.step(i, h, x_current, u, w)
        dynamics_params = integrator.jacobian(i, h, x_current, u, w)
        Sigma_current = propagateEllipsoid(Sigma_current, Sigmaw, dynamics_params, feedback_gains[i])
    # Terminal
    if xds is not None:
        x_terminal = xds[-1]
    elif xd_N is not None:
        x_terminal = xd_N
    else:
        x_terminal = None

    if x_terminal is not None:
        x_diff = x_current - x_terminal
        residual_list.append(np.dot(Qfsqrt,x_diff))
    for  obstacle in obstacles:
        residual_list.append(obstacleResidual(Sigma_current, x_current, obstacle, projection_matrix))
    return np.hstack(residual_list)
