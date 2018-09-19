#!/usr/bin/env python
import numpy as np

def getNominalDynamics(u, h, N, x0, integrator):
    xs= np.empty((N+1, x0.size))
    xs[0] = x0
    us = u.reshape(N, -1)
    w = np.zeros_like(x0)
    for i, u in enumerate(us):
        xs[i+1] = integrator.step(i, h, xs[i], u, w)
    return xs, us

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
