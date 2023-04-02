import numpy as np
import itertools
import scipy.spatial as sp
from resamvo.voronoi_multi import voronoi_cell_points_multi

def in_box(points_original, bounding_box):
    points = points_original.copy()
    return np.all(np.logical_and(bounding_box[:,0] <= points, points <= bounding_box[:,1]), axis=1)

def voronoi_finite_cell_points(points_original, points_target,bounding_box):
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
    
def voronoi_resample_num_ratio(source_df, target_df, match_prop):
    source_df_copy = source_df.copy()
    target_df_copy = target_df.copy()
    source_prop = source_df.loc[:,match_prop].values
    target_prop = target_df.loc[:,match_prop].values
    n_source, n_dim_source = source_prop.shape
    n_target, n_dim_target = target_prop.shape
    assert n_dim_source == n_dim_target
    # Compute bounding box
    bounding_box_mins = source_df_copy.min()[match_prop].values.reshape(n_dim_source,1)
    bounding_box_maxs = source_df_copy.max()[match_prop].values.reshape(n_dim_source,1)
    bounding_box =np.concatenate((bounding_box_mins, bounding_box_maxs), axis=1)
    # Compute nums in Voronoi cells
    cell_target_nums = voronoi_finite_cell_points(source_prop, target_prop,bounding_box)
    return cell_target_nums