#!/usr/bin/env python
import numpy as np
from ellipsoid import Ellipsoid

def projectEllipsoid(T, ellipsoid):
    L = ellipsoid.R*ellipsoid.S
    U, S, V = np.linalg.svd(np.dot(T, L))
    return Ellipsoid(U, S)
