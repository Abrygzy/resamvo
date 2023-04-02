import numpy as np
import itertools
import scipy.spatial as sp
from voronoi_multi import voronoi_cell_points_multi

def in_box(points_original, bounding_box):
    points = points_original.copy()
    return np.all(np.logical_and(bounding_box[:,0] <= points, points <= bounding_box[:,1]), axis=1)
def voronoi_finite_cell_points(points_original, bounding_box, points_target):
    points = points_original.copy()
    n_points, dim_points = points.shape
    in_box_index = in_box(points, bounding_box)
    # Mirror points
    points_center = points[in_box_index]
    points_all = points_center.copy()
    for i in range(dim_points):
        points_mirror_low = points_center.copy()
        points_mirror_high = points_center.copy()
        points_mirror_low[:,i] = bounding_box[i,0] - (points_mirror_low[:,i] - bounding_box[i,0])
        points_mirror_high[:,i] = bounding_box[i,1] + (bounding_box[i,1] - points_mirror_high[:,i])
        points_all = np.concatenate((points_all, points_mirror_low,points_mirror_high), axis=0)
    # Compute Voronoi
    vor = sp.Voronoi(points_all)
    filtered_points = points_center
    filtered_regions = list(map(vor.regions.__getitem__, vor.point_region[:n_points]))
    filtered_regions_flat = list(itertools.chain.from_iterable(filtered_regions))
    if (-1 in filtered_regions_flat):
        print("Warning: Some cells are open")
        print("Number of open cells: ", filtered_regions_flat.count(-1))
        return None
    cell_vertices = [vor.vertices[filtered_regions[i],:] for i in range(n_points)]
    cell_sdss_nums = voronoi_cell_points_multi(cell_vertices,points_target)
    return cell_sdss_nums