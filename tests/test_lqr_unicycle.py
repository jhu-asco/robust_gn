#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from robust_gnn.lqr_update import lqrUpdateGains
from robust_gnn.sampling import getNominalDynamics, sampleTrajectory
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.dynamics import UnicycleDynamics

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    dynamics = UnicycleDynamics()
    integrator = EulerIntegrator(dynamics)
    N = 30
    h = 0.1
    x0 = np.array([1,1,0])
    # Q, R, Qf
    cost_gains = [20*np.diag([1,1,1]), np.diag([0.1, 0.1]), 20*np.diag([2,2,2])]
    feedback_gains = np.empty((N, 2, 3))
    u = np.ravel(np.vstack((np.ones(N), np.hstack((np.ones(int(0.5*N)), -np.ones(int(N-0.5*N)))))).T)
    lqrUpdateGains(u, h, N, x0, cost_gains, integrator, feedback_gains)
    xds, uds = getNominalDynamics(u, h, N, x0, integrator)
    x0_perturb = x0 + np.array([0.1, 0.2, 0.2])
    xs, us = sampleTrajectory(x0_perturb, xds, uds, h, integrator, feedback_gains)
    plt.figure()
    plt.plot(xds[:,0], xds[:,1], 'r')
    plt.plot(xs[:,0], xs[:,1], 'b')
    plt.quiver(xs[:-1,0], xs[:-1,1], us[:,0]*np.cos(xs[:-1,2]), us[:,0]*np.sin(xs[:-1,2]), color='b', width=3, units='dots')
    plt.quiver(xds[:-1,0], xds[:-1,1], uds[:,0]*np.cos(xds[:-1,2]), uds[:,0]*np.sin(xds[:-1,2]), color='r', width=3, units='dots')
    plt.legend(['Desired', 'Feedback'])
    plt.show()
