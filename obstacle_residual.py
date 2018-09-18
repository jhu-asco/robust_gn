#!/usr/bin/env python
import numpy as np
from project_ellipsoid import projectEllipsoid

class Obstacle(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

def obstacleConstraint(Sigma_i, x_i, obstacle, T):
    proj_diff = np.dot(T, x_i) - obstacle.center
    Sigma_proj = projectEllipsoid(T, Sigma_i)
    rotated_x = np.dot(Sigma_proj.R.T, proj_diff)
    rotated_x_scaled = rotated_x/(Sigma_proj.S + obstacle.radius)
    constraint_distance = np.sum(np.square(rotated_x_scaled)) - 1.0
    return constraint_distance

def obstacleResidual(*args, **kwargs):
    return min(obstacleConstraint(*args, **kwargs), 0.0)
