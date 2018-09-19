#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from robust_gnn.lqr_update import lqrUpdateGains
from robust_gnn.gn_residual import residual
from robust_gnn.obstacle_residual import Obstacle
from robust_gnn.ellipsoid import Ellipsoid
from robust_gnn.sampling import getNominalDynamics, sampleTrajectory
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.dynamics import LinearDynamics

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    dynamics = LinearDynamics((np.array([[1,0],[0,1]]), np.array([[0],[1]])))
    integrator = EulerIntegrator(dynamics)
    N = 10
    h = 0.1
    x0 = np.array([0,0])
    Sigma0 = Ellipsoid(np.eye(2), np.array([0.1, 0.1]))
    Sigmaw = Ellipsoid(np.eye(2), np.array([0, 0]))
    # Q, R, Qf
    cost_gains = [np.diag([1,1]), np.diag([0.1]), np.diag([2,2])]
    cost_sqrt_gains = [np.sqrt(gain) for gain in cost_gains] # Not doing matrix sqrt since diagonal
    # Get feedback gains from LQR
    feedback_gains = np.empty((N, 1, 2))
    u0 = np.zeros(N)
    lqrUpdateGains(u0, h, N, x0, cost_gains, integrator, feedback_gains)
    # Obstacles
    obstacles = [Obstacle(np.array([1]), 0.5)]
    projection_matrix = np.array([[0, 1]])
    # Desired
    xd_N = np.array([1,0])
    xds = None
    uds = np.zeros((N,1))
    # Perform GN
    args = (h, N, x0, Sigma0, Sigmaw, xds, uds, integrator, cost_sqrt_gains, feedback_gains, obstacles, projection_matrix, xd_N)
    res_vec = residual(u0, *args)
    print res_vec
    # Perform GN now
    # Plotting
    
