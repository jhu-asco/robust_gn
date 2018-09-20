#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def plot2DEllipse(ax, center, radii, R, color='k'):
    theta = np.arctan2(R[1, 0], R[0, 0])
    e = Ellipse(center, 2*radii[0], 2*radii[1], theta, fill=False,
                ec=color)
    ax.add_patch(e)
    
def plotEllipsoid(self, center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid! Copied from ellipsoid tool"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)


        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
    
    if make_ax:
        plt.show()
        plt.close(fig)
        del fig

def plotObstacleArray(ax, obs_array):
    for obstacle in obs_array:
        if len(obstacle.center) == 2:
            radii = obstacle.radius*np.array([1, 1])
            plot2DEllipse(ax, obstacle.center, radii, np.eye(2), 'k')
        elif len(obstacle.center) == 3:
            radii = obstacle.radius*np.array([1, 1, 1])
            plotEllipsoid(obstacle.center, radii, np.eye(3), ax=ax,
                          cageColor='k')
        else:
            print "Cannot plot obstacles of dimensions other than 2, 3"

def plotEllipseArray(ax, ellipse_array):
    for ellipse in ellipse_array:
        if np.linalg.det(ellipse.R) < 0:
            ellipse.R[:, 0] = -1*ellipse.R[:,0]
        if len(ellipse.S) == 2:
            plot2DEllipse(ax, ellipse.center, ellipse.S, ellipse.R, 'b')
        elif len(ellipse.S) == 3:
            plotEllipsoid(ellipse.center, ellipse.S, ellipse.R, ax=ax,
                          cageColor='b')
        else:
            print "Cannot plot ellipse of dimensions other than 2, 3"
