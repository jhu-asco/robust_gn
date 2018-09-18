#!/usr/bin/env python
import numpy as np

def lqr_update_gains(u_in, h, N, x0, cost_gains, integrator, feedback_gains):
    us = u_in.reshape(N, -1)
    w = np.zeros_like(x0)
    dynamic_param_list = []
    x_current = x0
    # Forward Pass to get dynamics
    for i, u in enumerate(us):
        dynamic_param_list.append(integrator.jacobian(i, h, x_current, u, w))
        x_current = integrator.step(i, h, x_current, u, w)
    # Backward Pass
    Q, R, Qf = cost_gains
    P = Qf
    for i in range(N-1,-1,-1):
        A, B,_ = dynamic_param_list[i]
        BTP = np.dot(B.T, P)
        ATP = np.dot(A.T, P)
        R_bar = R + np.dot(BTP, B)
        # Update feedback gains
        feedback_gains[i] = -np.linalg.solve(R_bar, np.dot(BTP,A))
        A_bar = A + np.dot(B, feedback_gains[i])
        #Update P value
        P = Q + np.dot(ATP, A_bar)
