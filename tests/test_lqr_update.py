#!/usr/bin/env python
from robust_gnn.lqr_update import lqrUpdateGains
from optimal_control_framework.discrete_integrators import EulerIntegrator
from optimal_control_framework.dynamics import LinearDynamics, UnicycleDynamics
import unittest
import numpy as np
import numpy.testing as np_testing

class TestLQRUpdate(unittest.TestCase):
    def test_lqr_linear_system(self):
        dynamics = LinearDynamics((np.array([[0,1],[0,0]]), np.array([[0],[1]])))
        integrator = EulerIntegrator(dynamics)
        N = 10
        h = 0.1
        x0 = np.array([1,1])
        # Q, R, Qf
        cost_gains = [np.diag([1,1]), np.diag([1]), np.diag([2,2])]
        feedback_gains = np.empty((N, 1, 2))
        lqrUpdateGains(np.random.sample(N), h, N, x0, cost_gains, integrator,feedback_gains)
        np_testing.assert_array_less(feedback_gains[:-1, :, :], np.zeros_like(feedback_gains)[:-1, :, :])

    def test_lqr_unicycle(self):
        dynamics = UnicycleDynamics()
        integrator = EulerIntegrator(dynamics)
        N = 10
        h = 0.1
        x0 = np.array([1,1,0])
        # Q, R, Qf
        cost_gains = [np.diag([1,1,1]), np.diag([0.5, 1.0]), np.diag([2,2,2])]
        feedback_gains = np.empty((N, 2, 3))
        u = np.ravel(np.vstack((np.ones(N), np.hstack((np.ones(int(0.5*N)), -np.ones(int(N-0.5*N)))))).T)
        lqrUpdateGains(u, h, N, x0, cost_gains, integrator,feedback_gains)
