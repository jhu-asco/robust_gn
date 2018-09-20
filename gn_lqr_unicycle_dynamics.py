#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from robust_gnn.plot_ellipse import plotEllipseArray, plotObstacleArray
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
    us0 = np.zeros((N,2))
    Sigma0 = Ellipsoid(np.eye(n), np.array([0.1, 0.1, 0.01]))
    Sigmaw = Ellipsoid(np.eye(n), np.array([0, 0, 0]))
    integrator = EulerIntegrator(dynamics)
    # Q, R, Qf
    cost_gains = [np.diag([0,0,0]), np.diag([2, 0.05]), 200*np.diag([2, 2, 1])]
    # Obstacles
    obstacles = [Obstacle(np.array([0.5, 0.1]), 0.1)]
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0]])
    ko_sqrt = 20
    # Desired
    xds = None
    xd_N = np.array([1,0.2,0])
    v_average = np.linalg.norm(xd_N[:2] - x0[:2])/(N*h)
    print "V_average: ", v_average
    uds = np.zeros((N,m))
    uds[:, 0] = v_average
    us0[:,0] = v_average
    u0 = us0.ravel()
    # Perform GN with iter callback
    # Run algo and get outputs
    out = gn_lqr_algo(N, h, x0, u0, Sigma0, Sigmaw, integrator, cost_gains,
                      obstacles, projection_matrix, ko_sqrt, xds, uds, xd_N)
    print "xs_opt: ", out.xs_opt
    # Plotting
    plt.figure(1)
    ax = plt.gca()
    ax.set_aspect(aspect='equal')
    ax.set_ylim(-0.2, 0.5)
    ax.plot(out.xs_opt[:,0], out.xs_opt[:,1], 'b')
    ax.plot(xd_N[0], xd_N[1], 'r*')
    plt.legend(['Reference', 'Desired'])
    plotEllipseArray(ax, out.Sigma_out)
    plotObstacleArray(ax, obstacles)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.figure()
    ts = np.arange(N)*h
    plt.plot(ts, out.us_opt[:,0])
    plt.plot(ts, out.us_opt[:,1])
    plt.plot(ts, out.xs_opt[:-1, 2])
    plt.legend(['Velocity (m/s)', 'Angular rate (rad/s)', 'Angle (rad)'])
    plt.xlabel('Time')
    plt.show()
