from robust_gnn.project_ellipsoid import projectEllipsoid
from robust_gnn.ellipsoid import Ellipsoid
import numpy as np
import numpy.testing as np_testing
import unittest
import tf.transformations as tf

class TestProjectEllipsoid(unittest.TestCase):
    def testUnRotatedProjection(self):
        # R, S
        ellipsoid = Ellipsoid(np.eye(3), np.array([4,3,2]))
        T = np.array([[0,1,0],[0,0,1]])
        projected_ellipsoid = projectEllipsoid(T, ellipsoid)
        np_testing.assert_almost_equal(projected_ellipsoid.S, np.array([3,2]))
        np_testing.assert_almost_equal(projected_ellipsoid.R, np.array([[1,0],[0,1]]))

    def testRotatedProjection(self):
        # R, S
        ellipsoid = Ellipsoid(tf.euler_matrix(np.pi/3, np.pi/4, np.pi/6, 'rzyx')[:3,:3], np.array([4,5,6]))
        T = np.array([[0,1,0.5],[0.2,0.3,1]]) # Some linear projection
        projected_ellipsoid = projectEllipsoid(T, ellipsoid)
        N = 1000
        samples = np.random.sample((3, N))
        samples = samples/np.linalg.norm(samples, axis=0)
        for sample in samples.T:
            sample  = np.dot(ellipsoid.R, ellipsoid.S*sample)
            proj_sample = np.dot(T, sample)
            # Verify constraint
            rot_proj = np.dot(projected_ellipsoid.R.T, proj_sample)
            rot_proj_scaled = rot_proj/projected_ellipsoid.S
            self.assertLess(np.linalg.norm(rot_proj_scaled),1.0)

if __name__=="__main__":
    unittest.main()
