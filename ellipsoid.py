#!/usr/bin/env python

class Ellipsoid(object):
    """
    Define the rotation matrix, principal axes and
    center (optional)
    """
    def __init__(self, R, S, center=None):
        self.R = R
        self.S = S
        self.center = center
