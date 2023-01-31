#!/usr/bin/env python

"""discrete_curve.py file details"""

__authors__      = "Jax Burd & Abhijith Atreya"
__course__       = "UCSB ECE 594N: Geometric Machine Learning for Biomedical Imaging and Shape Analysis"
__professor__    = "Nina Miolane"
__deadline__     = "Thursday, 04/21/2022"

#-------------------------------------------------------------------------------------------------------------------

# IMPORTS
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import geomstats
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves

# CLASS
class DiscreteCurveViz:
    r"""Space of discrete curves sampled at points in ambient_manifold.
    Each individual curve is represented by a 2d-array of shape `[
    n_sampling_points, ambient_dim]`. A Batch of curves can be passed to
    all methods either as a 3d-array if all curves have the same number of
    sampled points, or as a list of 2d-arrays, each representing a curve.
    Parameters
    ----------
    curve_dimension : Manifold
        Manifold in which curves take values.
        
    param_curves_list: list
        List in which each element is a lamba function representing each parameterized curve.
        
    n_sampling_points : int
        Number of sampling points to applied to discretize the curves
    
    Attributes
    ----------
    dim : Manifold
        Manifold in which curves take values.
        
    param_curves : list
        List in which each element is a lamba function representing each parameterized curve.
        
    sampling_points : list
        List of sampling point values to pass through the parameterized curve functions for each curve.
        
    curve_points: list
        List of resulting points for each curve when their respective sampling points are applied to their curve function.

    """
    def __init__(self, curve_dimension, param_curves_list, sampling_points):
        self.dim = curve_dimension
        self.param_curves = param_curves_list
        self.sampling_points = sampling_points
        self.n = len(sampling_points[0])
        self.curve_points = self.set_curves()
    
    def set_curves(self):
        """ Internal helper function to pass sampling points through each curve function"""
        
        curves = []
        
        for i, p_curve in enumerate(self.param_curves):
            curves.append(p_curve(self.sampling_points[i]))
            
        return curves
    
    def resample(self, adjusted_sampling_points):
        """
        
        """
        self.sampling_points = adjusted_sampling_points
        curves = []
        
        for i, p_curve in enumerate(self.param_curves):
            curves.append(p_curve(self.sampling_points[i]))
        
        self.n = len(list(adjusted_sampling_points))
        
        self.curves_points = curves
        
    def plot_3Dcurves(self, linestyles, labels, title):
        """Create plots of given set of curves in 3D graph space.
        Parameters
            ----------
            linestyles : array-like, string elements
                Matpotlib linestyles to apply to respective curves.

            labels :  array-like, string elements
                Labels for axes on the plot.

            title : string
                Title of the plot.
          """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for i,curve in enumerate(self.curve_points):
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], linestyles[i], linewidth=2)
        ax.set_xlabel(labels[0]);
        ax.set_ylabel(labels[1]);
        ax.set_zlabel(labels[2]);
        ax.set_title(title)

    def plot_geodesic(self, n_times, inital_index, end_index, linestyles, labels, title):
        """Create plots of geodesic between two chosen curves.
        Parameters
            ----------
            n_times : int
                Number of geodesic curves to plot inbetween

            inital_index : int
                Index of the starting curve.

            end_index : int
                Index of the end curve.

            linestyles : array-like, string elements
                Matpotlib linestyles to apply to respective curves.

            labels :  array-like, string elements
                Labels for axes on the plot.

            title : string
                Title of the plot.

            """
        geod_fun = self.dim.srv_metric.geodesic(
            initial_point=self.curve_points[inital_index], end_point=self.curve_points[end_index]
            )
        
        times = np.linspace(0.0, 1.0, n_times)
        geod = geod_fun(times)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        plt.figure(figsize=(10, 10))
        ax.plot(geod[0, :, 0], geod[0, :, 1], geod[0, :, 2], linestyles[0])
        
        for i in range(1, n_times - 1):
            ax.plot(geod[i, :, 0], geod[i, :, 1], geod[i, :, 2], linestyles[1])
            
        
        ax.plot(geod[-1, :, 0], geod[-1, :, 1], geod[-1, :, 2], linestyles[2])
        
        ax.set_title(title)
        ax.set_xlabel(labels[0]);
        ax.set_ylabel(labels[1]);
        ax.set_zlabel(labels[2]);
        
    def plot_geodesic_net(self, n_times, inital_index, end_index, linestyles, labels, title, view_init):
        """ Creates a plot of geodesics between two chosen curves in a wireframe style.
        Parameters
            ----------
            n_times : int
                Number of geodesic curves to plot inbetween

            inital_index : int
                Index of the starting curve.

            end_index : int
                Index of the end curve.

            linestyles : array-like, string elements
                Matpotlib linestyles to apply to respective curves.

            labels :  array-like, string elements
                Labels for axes on the plot.

            title : string
                Title of the plot.
        
        """
        geod_fun = self.dim.srv_metric.geodesic(
            initial_point=self.curve_points[inital_index], end_point=self.curve_points[end_index]
            )
        
        times = np.linspace(0.0, 1.0, n_times)
        geod = geod_fun(times)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        plt.figure(figsize=(10, 10))
        
        ax.plot3D(geod[0, :, 0], geod[0, :, 1], geod[0, :, 2], linestyles[0], linewidth=2)
        for i in range(1, n_times - 1):
            ax.plot3D(geod[i, :, 0], geod[i, :, 1], geod[i, :, 2], linestyles[1],linewidth=1)
        for j in range(self.n):
            ax.plot3D(geod[:, j, 0], geod[:, j, 1], geod[:, j, 2], linestyles[1], linewidth=1)
        ax.plot3D(geod[-1, :, 0], geod[-1, :, 1], geod[-1, :, 2], linestyles[2], linewidth=2)
            
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        
        if view_init:
            ax.view_init(view_init[0],view_init[1])
            fig

    def plot_parallel_transport(self, n_times, sampling_point_index, inital_index, end_index,
                                linestyles, labels, title, view_init):
        """
        Highlights a certain line along a geodesic between curves in red
        Parameters
            ----------
            n_times : int
                Number of geodesic curves to plot inbetween
                
            sampling_point_index : int
                Index of sampling point on both curves to highlight parallel transport

            inital_index : int
                Index of the starting curve.

            end_index : int
                Index of the end curve.

            linestyles : array-like, string elements
                Matpotlib linestyles to apply to respective curves.

            labels :  array-like, string elements
                Labels for axes on the plot.

            title : string
                Title of the plot.
                
            view_init : array-like
                List of elevation arguement and rotation argument of plot3D view angle
        
        """
        geod_fun = self.dim.srv_metric.geodesic(
            initial_point=self.curve_points[inital_index], end_point=self.curve_points[end_index]
            )
        
        times = np.linspace(0.0, 1.0, n_times)
        geod = geod_fun(times)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        plt.figure(figsize=(10, 10))
        
        ax.plot3D(geod[0, :, 0], geod[0, :, 1], geod[0, :, 2], linestyles[0], linewidth=2)
        ax.plot3D(geod[0, sampling_point_index, 0],
                  geod[0, sampling_point_index, 1],
                  geod[0, sampling_point_index, 2], 'or', linewidth=2)
        
        for i in range(1, n_times - 1):
            ax.plot3D(geod[i, :, 0], geod[i, :, 1], geod[i, :, 2], linestyles[1],linewidth=1)
        for j in range(self.n):
            if j is sampling_point_index:
                ax.plot3D(geod[:, j, 0], geod[:, j, 1], geod[:, j, 2], 'r', linewidth=2)
                
            else:
                ax.plot3D(geod[:, j, 0], geod[:, j, 1], geod[:, j, 2], linestyles[1], linewidth=1)
            
        ax.plot3D(geod[-1, :, 0], geod[-1, :, 1], geod[-1, :, 2], linestyles[2], linewidth=2)
        ax.plot3D(geod[-1, sampling_point_index, 0],
                  geod[-1, sampling_point_index, 1],
                  geod[-1, sampling_point_index, 2], 'or', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        
        if view_init:
            ax.view_init(view_init[0],view_init[1])
            fig
            
# TESTING
def main():
    # Example Test
    
    R3 = Euclidean(dim=3)
    dc = DiscreteCurves(ambient_manifold=R3)
    param_curves_list = [lambda x: np.array([np.cos(x), np.sin(x), x]),
                             lambda x: np.array([np.sin(x), np.cos(x), x])]
    sampling_points = [np.linspace(0, 2 * np.pi, 10), np.linspace(0, 2 * np.pi, 5)]
    dcv = DiscreteCurveViz(curve_dimension = dc, param_curves_list = param_curves_list, sampling_points = sampling_points)
    linestyles = ['r-', 'b-']
    labels = ['x', 'y', 'z']
    title = '3D Curve Plot'
    dcv.plot_3Dcurves(linestyles, labels, title)

    return 0
    

if __name__ == "__main__":
    main()
