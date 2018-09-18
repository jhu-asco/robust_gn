#!/usr/bin/env python
import numpy as np
from ellipsoid import Ellipsoid

def projectEllipsoid(T, ellipsoid):
    L = ellipsoid.R*ellipsoid.S
    U, S, V = np.linalg.svd(np.dot(T, L))
    if ellipsoid.center is not None:
        center = np.dot(T, ellipsoid.center)
    else:
        center = None
    return Ellipsoid(U, S, center)
