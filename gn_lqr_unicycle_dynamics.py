#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from robust_gnn.gn_lqr_algo import gn_lqr_algo
from robust_gnn.obstacle_residual import Obstacle
from robust_gnn.ellipsoid import Ellipsoid
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.dynamics import UnicycleDynamics


if __name__ == "__main__":
    dynamics = UnicycleDynamics()
    N = 10
    h = 0.1
    n = 3
    m = 2
    x0 = np.array([0,0,0])
    us = np.zeros((N,2))
    us[:,0] = 1
    u0 = us.ravel()
    Sigma0 = Ellipsoid(np.eye(n), np.array([0.1, 0.1, 0.01]))
    Sigmaw = Ellipsoid(np.eye(n), np.array([0, 0, 0]))
    integrator = EulerIntegrator(dynamics)
    # Q, R, Qf
    cost_gains = [np.diag([1,1,1]), np.diag([0.5, 0.1]), 100*np.diag([2, 2, 1])]
    # Obstacles
    obstacles = [Obstacle(np.array([0.5, 0.1]), 0.15)]
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0]])
    ko_sqrt = 2
    # Desired
    xds = None
    uds = np.zeros((N,1))
    xd_N = np.array([1,0.2,0])
    # Perform GN with iter callback
    # Run algo and get outputs
    out = gn_lqr_algo(N, h, x0, u0, Sigma0, Sigmaw, integrator, cost_gains,
                      obstacles, projection_matrix, ko_sqrt, xds, uds, xd_N)
    radii = np.stack([Sigma.S for Sigma in out.Sigma_out], axis=0)
    print "xs_opt: ", out.xs_opt
    print "Sigma radii: "
    print radii
    print radii.shape
    # Plotting
