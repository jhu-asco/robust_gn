#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from robust_gnn.gn_lqr_algo import gn_lqr_algo
from robust_gnn.obstacle_residual import Obstacle
from robust_gnn.ellipsoid import Ellipsoid
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.dynamics import LinearDynamics

if __name__ == '__main__':
    dynamics = LinearDynamics((np.array([[0,1],[0,0]]), np.array([[0],[1]])))
    N = 10
    h = 0.1
    u0 = np.zeros(N)
    x0 = np.array([0,0])
    Sigma0 = Ellipsoid(np.eye(2), np.array([0.1, 0.1]))
    Sigmaw = Ellipsoid(np.eye(2), np.array([0, 0]))
    integrator = EulerIntegrator(dynamics)
    # Q, R, Qf
    cost_gains = [np.diag([1,1]), np.diag([0.1]),
                  100*np.array([[2,0.2],[0.2, 2]])]
    # Obstacles
    obstacles = [Obstacle(np.array([1]), 0.5)]
    projection_matrix = np.array([[0, 1]])
    ko_sqrt = 2
    # Desired
    xds = None
    uds = np.zeros((N,1))
    xd_N = np.array([0.5,0])
    # Run algo and get outputs
    out = gn_lqr_algo(N, h, x0, u0, Sigma0, Sigmaw, integrator, cost_gains,
                      obstacles, projection_matrix, ko_sqrt, xds, uds, xd_N)
    radii = np.stack([Sigma.S for Sigma in out.Sigma_out], axis=0)
    print radii.shape
    print "Stdev projected on velocity: ", radii
    print  "Xs_opt: "
    print out.xs_opt
    print "xd_N: ", xd_N
    print "res: \n", out.gn_out.fun

