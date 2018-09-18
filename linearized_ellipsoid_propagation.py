#!/usr/bin/env python
# Propagate the ellipsoids linearly
import numpy as np
from scipy.linalg import block_diag
from ellipsoid import Ellipsoid
from project_ellipsoid import projectEllipsoid

def propagateEllipsoid(Sigma_i, Sigma_w_i, dynamics_params, feedback_gain):
    """
    Given an ellipsoid at 
    """
    A, B, G = dynamics_params
    Abar = A + np.dot(B, feedback_gain)
    T = np.hstack((Abar, G))
    Sigma_full = Ellipsoid(block_diag(Sigma_i.R, Sigma_w_i.R),
                           np.hstack((Sigma_i.S, Sigma_w_i.S)))
    return projectEllipsoid(T, Sigma_full)
