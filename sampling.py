#!/usr/bin/env python
import numpy as np
from robust_gnn.linearized_ellipsoid_propagation import propagateEllipsoid
from robust_gnn.project_ellipsoid import projectEllipsoid

def getNominalDynamics(u, h, N, x0, integrator):
    xs= np.empty((N+1, x0.size))
    xs[0] = x0
    us = u.reshape(N, -1)
    w = np.zeros_like(x0)
    for i, u in enumerate(us):
        xs[i+1] = integrator.step(i, h, xs[i], u, w)
    return xs, us

def getNominalEllipsoids(h, xs, us, Sigma0, Sigmaw, projection_matrix, feedback_gains, integrator):
    Sigma_out = [projectEllipsoid(projection_matrix, Sigma0)]
    Sigma_current = Sigma0
    w = np.zeros_like(xs[0])
    for i, u in enumerate(us):
        dynamics_params = integrator.jacobian(i, h, xs[i], u, w)
        Sigma_current = propagateEllipsoid(Sigma_current, Sigmaw, dynamics_params, feedback_gains[i])
        Sigma_out.append(projectEllipsoid(projection_matrix, Sigma_current))
    return Sigma_out

def sampleTrajectory(x0, xds, uds, h, integrator, feedback_gains):
    xs = np.empty_like(xds)
    us = np.empty_like(uds)
    xs[0] = x0
    w = np.zeros_like(x0)
    for i, u in enumerate(uds):
        delta_x = xs[i] - xds[i]
        u_in = u + np.dot(feedback_gains[i], delta_x)
        us[i] = u_in
        xs[i+1] = integrator.step(i, h, xs[i], u_in, w)
    return xs, us
