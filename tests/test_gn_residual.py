#!/usr/bin/env python
from robust_gnn.sampling import getNominalDynamics
from robust_gnn.ellipsoid import Ellipsoid
from robust_gnn.gn_residual import residual
from robust_gnn.obstacle_residual import Obstacle
from optimal_control_framework.dynamics import LinearDynamics
from optimal_control_framework.discrete_integrators import EulerIntegrator
import unittest
import numpy as np
import numpy.testing as np_testing


class TestGnResidual(unittest.TestCase):
    def testResidualNoNoise(self):
        dynamics = LinearDynamics((np.array([[1,0],[0,1]]), np.array([[1, 0],[0, 1]])))
        integrator = EulerIntegrator(dynamics)
        Sigma0 = Ellipsoid(np.eye(2), np.array([0, 0]))
        Sigmaw = Ellipsoid(np.eye(2), np.array([0, 0]))
        N = 10
        h = 0.1
        x0 = np.array([1,1])
        # Q, R, Qf
        cost_gains = [np.diag([1,1]), np.diag([1, 1]), np.diag([2,2])]
        cost_sqrt_gains = [np.sqrt(gain) for gain in cost_gains]
        # Obstacles
        obstacles = [Obstacle(np.array([2,2]), 1.2), Obstacle(np.array([3,3]), 0.2)]
        # Desired States, Controls:
        xds = None
        uds = np.zeros((N, 2))
        # Controls:
        us = np.zeros((N, 2))
        us[:,0] = 1.0
        projection_matrix = np.eye(2)
        feedback_gains = np.zeros((N, 2, 2))
        residual_vector = residual(us.ravel(), h, N, x0, Sigma0, Sigmaw, xds, uds, integrator, cost_sqrt_gains, feedback_gains, obstacles, projection_matrix)
        Rsqrt = cost_sqrt_gains[1]
        # Nominal dynamics
        xs,_ = getNominalDynamics(us.ravel(), h, N, x0, integrator)
        index = 0
        for i in range(N):
            np_testing.assert_almost_equal(residual_vector[index:(index+2)], np.dot(Rsqrt, us[i]))
            index = index + 2
            for obstacle in obstacles:
                obs_distance = np.linalg.norm(xs[i] - obstacle.center)
                if obs_distance > obstacle.radius:
                    np_testing.assert_equal(residual_vector[index:(index+2)], np.zeros(2))
                else:
                    self.assertGreater(np.linalg.norm(residual_vector[index:(index+2)]), 0)
                index = index + 2
        # Terminal
        for obstacle in obstacles:
            obs_distance = np.linalg.norm(xs[-1] - obstacle.center)
            if obs_distance > obstacle.radius:
                np_testing.assert_equal(residual_vector[index:(index+2)], np.zeros(2))
            else:
                self.assertGreater(np.linalg.norm(residual_vector[index:(index+2)]), 0)
            index = index + 2
        self.assertEqual(index, residual_vector.size)
