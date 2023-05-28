#!/opt/anaconda3/envs/py311/bin/python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import scipy.spatial as sp
import numpy as np
import pandas as pd
eps = np.finfo(np.float32).eps

def test(x,y):
     return x**2 + y**2

# Calculate the volume of a Voronoi cell
def voronoi_cell_volume(cell_vertices): 
    return sp.ConvexHull(cell_vertices).volume


def voronoi_cell_volume_multi(cell_vertices_list): 
        pool = mp.Pool()
        return pool.map(voronoi_cell_volume, cell_vertices_list)


# Test the numbers of points in each Voronoi cell
def voronoi_cell_points_sigle(cell_vertices, target_points_original): 
    # cell_vertices: 构成一个cell的所有顶点的坐标，array (n_vertices, n_dim)
    # target_points_original: 所有的目标点的坐标，array (n_points, n_dim)
    target_points = target_points_original.copy()
    eps = np.finfo(np.float64).eps # 考虑到浮点数的误差，判断是不是在凸包内部的时候，用eps作为阈值
    # Initial Cut: remove target_points by the vertices of convex hull 
    low_boundaries = np.min(cell_vertices, axis=0)
    up_boundaries = np.max(cell_vertices, axis=0)
    initial_cut_ind = np.all(
        ((target_points>low_boundaries) & (target_points<up_boundaries)),axis=1)
    target_points_initial_cut = target_points[initial_cut_ind]
    # Cal points inside the convex hull of the Voronoi cell
    hull = sp.ConvexHull(cell_vertices)
    # A is shape (f, d) and b is shape (f, 1).
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    # If each point is inside the convex hull
        # The hull is defined as all points x for which Ax + b <= 0.
        # We compare to a small positive value to account for floating point issues.
        # Assuming x is shape (m, d), output is boolean shape (m,).
    contained_flag = np.all(np.asarray(target_points_initial_cut) @ A.T + b.T < eps, axis=1)
    return np.sum(contained_flag)
    # return target_points_initial_cut[contained_flag]

def voronoi_cell_points_multi(cell_vertices_list,target_points_original):
    target_points = target_points_original.copy()
    pool = mp.Pool()
    return pool.starmap(voronoi_cell_points_sigle, zip(cell_vertices_list, [target_points]*len(cell_vertices_list)))
