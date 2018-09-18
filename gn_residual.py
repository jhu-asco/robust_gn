#!/usr/bin/env python
import numpy as np
from linearized_ellipsoid_propagation import propagateEllipsoid
from obstacle_residual import obstacleResidual

def residual(u_in, h, N, x0, Sigma0, Sigma_w, xds, uds, integrator, cost_gains, feedback_gains, obstacles, projection_matrix):
    us = u_in.reshape(N, -1)
    w = np.zeros_like(x0)
    x_current = x0
    Sigma_current = Sigma0
    residual_list = []
    Q, R, Qf = cost_gains
    for i, u in enumerate(us):
        # Create residual
        x_diff = x_current - xds[i]
        u_diff = u - uds[i]
        residual_list = residual_list + [Q*x_diff, R*u_diff]
        for  obstacle in obstacles:
            residual_list.append(obstacleResidual(Sigma_current, x_current, obstacle, projection_matrix))
        x_current = integrator.step(i, h, x_current, u, w)
        dynamics_params = integrator.jacobian(i, h, x_current, u, w)
        Sigma_current = propagateEllipsoid(Sigma_current, Sigma_w, dynamics_params, feedback_gains[i])
    # Terminal
    x_diff = x_current - xds[-1]
    residual_list = residual_list + [Qf*x_diff]
    for  obstacle in obstacles:
        residual_list.append(obstacleResidual(Sigma_current, x_current, obstacle, projection_matrix))
    return np.hstack(residual_list)