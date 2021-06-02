import logging
from typing import NamedTuple
from typing import Tuple

import open3d as o3d
from scipy.spatial.transform import Rotation
from visualise import visualise_pointclouds
from visualise import VisualiseLevel

LOGGER = logging.getLogger(__name__)

PlaneT = Tuple[float, float, float, float]


PLANE_SEGMENTATION_RANSAC_DISTANCE_THRESHOLD = 0.03
PLANE_SEGMENTATION_RANSAC_ITERATIONS = 1000


class Planes(NamedTuple):
    floor: PlaneT
    wall: PlaneT


def pointcloud_find_planes(
    pcd: o3d.geometry.PointCloud,
    visualise_level: VisualiseLevel,
) -> Tuple[PlaneT, PlaneT, o3d.geometry.PointCloud]:
    """Find 2 planes with largest supports in the point cloud."""
    LOGGER.info("Finding 2 largest planes")

    plane1, inliers1 = pcd.segment_plane(
        ransac_n=3,
        distance_threshold=PLANE_SEGMENTATION_RANSAC_DISTANCE_THRESHOLD,
        num_iterations=PLANE_SEGMENTATION_RANSAC_ITERATIONS,
    )

    outlier1_cloud = pcd.select_by_index(inliers1, invert=True)

    plane2, inliers2 = outlier1_cloud.segment_plane(
        ransac_n=3,
        distance_threshold=PLANE_SEGMENTATION_RANSAC_DISTANCE_THRESHOLD,
        num_iterations=PLANE_SEGMENTATION_RANSAC_ITERATIONS,
    )

    LOGGER.debug(f"Plane 1 equation: {plane1[0]:.3f}x + {plane1[1]:.3f}y + {plane1[2]:.3f}z + {plane1[3]:.3f} = 0")
    LOGGER.debug(f"Plane 2 equation: {plane2[0]:.3f}x + {plane2[1]:.3f}y + {plane2[2]:.3f}z + {plane2[3]:.3f} = 0")

    plane1_cloud = pcd.select_by_index(inliers1)
    plane2_cloud = outlier1_cloud.select_by_index(inliers2)
    remaining_cloud = outlier1_cloud.select_by_index(inliers2, invert=True)

    if visualise_level >= VisualiseLevel.HIGH:
        plane1_cloud.paint_uniform_color([1.0, 0, 0])
        plane2_cloud.paint_uniform_color([0, 1.0, 0])

        visualise_pointclouds(plane1_cloud, plane2_cloud, remaining_cloud)

    return plane1, plane2, remaining_cloud


def distinguish_between_floor_and_wall_planes(plane1: PlaneT, plane2: PlaneT) -> Planes:
    """Identify which plane is the floor and which is the wall.

    Assuming iPad is nearly upright, choose the plane with largest y normal component as floor."""

    if plane1[1] > plane2[1]:
        return Planes(floor=plane1, wall=plane2)
    else:
        return Planes(floor=plane2, wall=plane1)


def orientate_pointcloud(
    pcd: o3d.geometry.PointCloud,
    planes: Planes,
    visualise_level: VisualiseLevel,
) -> o3d.geometry.PointCloud:
    """Rotate point cloud so floor is on xz plane and wall is on xy plane. Translate origin to floor."""
    LOGGER.info("Orientating pointcloud")

    # find rotation matrix
    align_mat, _ = Rotation.align_vectors([[0, 1, 0], [0, 0, 1]], [planes.floor[:3], planes.wall[:3]])

    pcd.rotate(align_mat.as_matrix(), [0, 0, 0])
    pcd.translate([0, planes.floor[3], 0])

    if visualise_level >= VisualiseLevel.HIGH:
        visualise_pointclouds(pcd)

    return pcd
