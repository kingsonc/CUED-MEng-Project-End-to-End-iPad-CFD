import logging

import numpy as np
import open3d as o3d
from scipy.stats import mode
from visualise import visualise_pointclouds
from visualise import VisualiseLevel


LOGGER = logging.getLogger(__name__)

OUTLIER_NUM_NEIGHBOURS = 20
OUTLIER_STD_RATIO = 0.8

DBSCAN_DENSITY_EPS = 0.01
DBSCAN_MIN_POINTS = 10


def crop_floor(pcd: o3d.geometry.PointCloud, visualise_level: VisualiseLevel) -> o3d.geometry.PointCloud:
    x_limit = 0.1
    y_limit = 0.5

    top_limit = 0.2
    bot_limit = 0.01

    bbox = [
        # bottom face
        [-x_limit, bot_limit, -y_limit],
        [x_limit, bot_limit, -y_limit],
        [x_limit, bot_limit, y_limit],
        [-x_limit, bot_limit, y_limit],

        # top face
        [-x_limit, top_limit, -y_limit],
        [x_limit, top_limit, -y_limit],
        [x_limit, top_limit, y_limit],
        [-x_limit, top_limit, y_limit],
    ]

    bbox_points = o3d.utility.Vector3dVector(bbox)
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(bbox_points)

    pcd = pcd.crop(oriented_bounding_box)

    if visualise_level >= VisualiseLevel.LOW:
        visualise_pointclouds(pcd)

    return pcd


def remove_pointcloud_outliers(
    pcd: o3d.geometry.PointCloud,
    visualise_level: VisualiseLevel,
    std_ratio: float = OUTLIER_STD_RATIO,
) -> o3d.geometry.PointCloud:
    LOGGER.info("Removing pointcloud outliers")

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NUM_NEIGHBOURS, std_ratio=std_ratio)

    if visualise_level >= VisualiseLevel.HIGH:
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)

        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        visualise_pointclouds(inlier_cloud, outlier_cloud)
        visualise_pointclouds(cl)

    return cl


def pointcloud_find_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    visualise_level: VisualiseLevel,
) -> o3d.geometry.PointCloud:
    """Find largest cluster of points in pointcloud using DBSCAN - Density-Based Spatial Clustering of
    Applications with Noise algorithm. Assume the largest cluster in the pointcloud is the object of interest."""
    LOGGER.info("Identifying pointcloud clusters")

    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_DENSITY_EPS, min_points=DBSCAN_MIN_POINTS))

    LOGGER.debug(f"Found {labels.max() + 1} clusters")

    most_freq_label = mode(labels).mode
    indices = np.where(labels == most_freq_label)[0]
    cluster = pcd.select_by_index(indices)

    if visualise_level >= VisualiseLevel.LOW:
        visualise_pointclouds(cluster)

    return cluster
