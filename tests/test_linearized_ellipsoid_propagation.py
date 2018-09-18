#!/usr/bin/env python
from robust_gnn.linearized_ellipsoid_propagation import propagateEllipsoid
from robust_gnn.ellipsoid import Ellipsoid
import numpy as np
import numpy.testing as np_testing
import unittest

class TestLinearizedPropagationEllipsoid(unittest.TestCase):
    def testEllipsoidPropagationStableDynamics(self):
        Sigma_i = Ellipsoid(np.eye(2), np.array([1,2]))
        Sigma_w_i = Ellipsoid(np.eye(2), np.array([0.1,0.01]))
        # A, B, G
        dynamics_params = [np.array([[1,0.1], [0, 0]]), np.array([[0],[0.1]]), np.eye(2)]
        feedback_gain = np.array([[-5, -5]])
        for i in range(50):
            Sigma_n = propagateEllipsoid(Sigma_i, Sigma_w_i, dynamics_params, feedback_gain)
            self.assertLess(np.linalg.norm(Sigma_n.S), np.linalg.norm(Sigma_i.S))
            Sigma_i = Sigma_n

if __name__=="__main__":
    unittest.main()
