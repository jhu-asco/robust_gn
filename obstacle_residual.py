#!/usr/bin/env python
import numpy as np
from project_ellipsoid import projectEllipsoid

class Obstacle(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

def obstacleResidual(Sigma_i, x_i, obstacle, T, clip=True):
    proj_diff = np.dot(T, x_i) - obstacle.center
    Sigma_proj = projectEllipsoid(T, Sigma_i)
    rotated_x = np.dot(Sigma_proj.R.T, proj_diff)
    S_expanded = (Sigma_proj.S + obstacle.radius)
    rotated_x_scaled = rotated_x/S_expanded
    norm_rotated_x_scaled = np.linalg.norm(rotated_x_scaled)
    if clip and norm_rotated_x_scaled >= 1.0:
        return np.zeros_like(rotated_x_scaled)
    elif norm_rotated_x_scaled < 1e-12:
        print "Warning! Passing through center of obstacle!"
        unit_vec = np.zeros_like(rotated_x_scaled)
        unit_vec[0] = 1
        return S_expanded*(rotated_x_scaled - unit_vec)
    else:
        unit_vec = rotated_x_scaled/norm_rotated_x_scaled
        delta_x = S_expanded*(rotated_x_scaled - unit_vec)
        return delta_x

def obstacleConstraint(Sigma_i, x_i, obstacle, T, clip=True):
    proj_diff = np.dot(T, x_i) - obstacle.center
    Sigma_proj = projectEllipsoid(T, Sigma_i)
    rotated_x = np.dot(Sigma_proj.R.T, proj_diff)
    S_expanded = (Sigma_proj.S + obstacle.radius)
    rotated_x_scaled = rotated_x/S_expanded
    norm_rotated_x_scaled = np.linalg.norm(rotated_x_scaled)
    if norm_rotated_x_scaled < 1e-12:
        unit_vec = np.zeros_like(rotated_x_scaled)
        unit_vec[0] = 1
    else:
        unit_vec = rotated_x_scaled/norm_rotated_x_scaled
    residual = obstacleResidual(Sigma_i, x_i, obstacle, T, clip)
    return np.dot(residual, unit_vec)
