from robust_gnn.obstacle_residual import Obstacle, obstacleConstraint,obstacleResidual
from robust_gnn.ellipsoid import Ellipsoid
import tf.transformations as tf
import numpy as np
import numpy.testing as np_testing
import unittest

class TestObstacleConstraint(unittest.TestCase):
    def testUnRotatedEllipsoid(self):
        obstacle = Obstacle(np.array([1,1]), 2.0)
        projection_matrix = np.array([[1,0,0], [0,1,0]])
        Sigma = Ellipsoid(np.eye(3), np.array([0.2, 0.1, 0.6]))
        # Inside
        constraint = obstacleConstraint(Sigma, np.zeros(3), obstacle, projection_matrix, clip=False)
        self.assertLess(constraint, 0)
        # Passing through center
        constraint = obstacleConstraint(Sigma, np.ones(3), obstacle, projection_matrix, clip=False)
        self.assertLess(constraint, 0)
        # Outside
        constraint = obstacleConstraint(Sigma, -1*np.ones(3), obstacle, projection_matrix, clip=False)
        self.assertGreater(constraint, 0)
        # Check residual with clipping
        residual = obstacleResidual(Sigma, -1*np.ones(3), obstacle, projection_matrix, clip=True)
        np_testing.assert_equal(residual, np.zeros_like(residual))
        # On boundary
        constraint = obstacleConstraint(Sigma, np.array([-1.2, 1, 0]), obstacle, projection_matrix, clip=False)
        self.assertEqual(constraint, 0)
        # On boundary2
        constraint = obstacleConstraint(Sigma, np.array([1, 3.1, 0]), obstacle, projection_matrix, clip=False)
        self.assertEqual(constraint, 0)

    def testRotatedEllipsoid(self):
        obstacle = Obstacle(np.array([1,1]), 2.0)
        projection_matrix = np.array([[0,1,0], [0,0,1]])
        R = tf.euler_matrix(np.pi/4, 0, 0, 'rzyx')[:3, :3]
        Sigma = Ellipsoid(R, np.array([0, 0.2, 0.2]))
        # Finally z should be 0.2, y should be 0.2/sqrt(2)
        constraint = obstacleConstraint(Sigma, np.array([0, -1, -1]), obstacle, projection_matrix, clip=False)
        self.assertGreater(constraint, 0)
        constraint = obstacleConstraint(Sigma, np.array([0, 1, 3.2]), obstacle, projection_matrix, clip=False)
        self.assertEqual(constraint, 0)
        constraint = obstacleConstraint(Sigma, np.array([0, -1-0.2/np.sqrt(2), 1]), obstacle, projection_matrix, clip=False)
        self.assertAlmostEqual(constraint, 0, 12)

if __name__=="__main__":
    unittest.main()
