import argparse
import logging
from datetime import datetime
from typing import List
from typing import Tuple

import algorithms
import matplotlib.pyplot as plt
import mesh
import numpy as np
import open3d as o3d
from optimisation.least_squares import least_squares
from optimisation.minimize import solver
from visualise import visualise_pointclouds
from visualise import VisualiseLevel

from visualise import plot_optimised


LOGGER = logging.getLogger(__name__)

CUT_HEIGHT = 0.075  # halfway along blade (15cm)
CUT_TOLERANCE = 0.15 * 0.15  # 15% thickness

EDGE_IDENTIFICATION_TOL = 0.01
EDGE_RELATIVE_MAX_POS = 0.2

plt.rc('font', size=24)
plt.rcParams["figure.figsize"] = (10, 7)


def post_process(
    pcd: o3d.geometry.PointCloud,
    visualise_level: VisualiseLevel = VisualiseLevel.HIGH,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], Tuple[float, float], List[Tuple[float, float, float]]]:
    LOGGER.info("Begin post processing")

    if visualise_level >= VisualiseLevel.LOW:
        visualise_pointclouds(pcd)

    # floor_noise_experiment(pcd)

    pcd = algorithms.crop_floor(pcd, visualise_level=visualise_level)

    cluster = algorithms.pointcloud_find_largest_cluster(pcd, visualise_level=visualise_level)

    cut = pointcloud_cut(cluster, cut_height=CUT_HEIGHT, visualise_level=visualise_level)
    cut, blade_start, blade_end = normalize_cut(cut, visualise_level=visualise_level)

    # # objective minimisation
    # solver(cut)

    upper_surface_coeffs, lower_surface_coeffs = least_squares(cut, visualise_level=visualise_level)
    aerofoil_mesh = mesh.aerofoil_mesh(upper_surface_coeffs, lower_surface_coeffs)

    LOGGER.info("Post processing complete")

    return upper_surface_coeffs, lower_surface_coeffs, blade_start, blade_end, aerofoil_mesh


def floor_noise_experiment(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """For experiment to measure standard deviation of noise.
    Crops walls and phone (marker) to leave flat floor.
    """
    points = np.asarray(pcd.points)
    points_cropped = points[
        (points[:, 0] < 0.5)
        & (points[:, 0] > -0.5)
        & (points[:, 2] > -0.5)
        & (points[:, 2] < 0.5)
        & ((points[:, 0] < -0.05) | (points[:, 0] > 0.05))
        & ((points[:, 2] > 0.1) | (points[:, 2] < -0.1))
    ]

    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points_cropped)
    visualise_pointclouds(pcd_tmp)

    z_points = points_cropped[:, 1]

    print(f'Number points: {len(z_points)}')
    print(f'Mean: {z_points.mean()}')
    print(f'Std: {z_points.std()}')

    plt.hist(z_points, bins=100)
    plt.xlabel("Height (m)")
    plt.ylabel("Frequency of points")
    plt.show()

    return pcd


def convert_to_pointcloud(points: List[List[float]]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def open_pointcloud(filename: str) -> o3d.geometry.PointCloud:
    LOGGER.info(f"Opening pointcloud file. {filename=}")
    return o3d.io.read_point_cloud(filename)


def save_pointcloud(pcd: o3d.geometry.PointCloud) -> None:
    filename = f'pcd-{datetime.now().isoformat()}.pcd'
    o3d.io.write_point_cloud(filename, pcd)
    LOGGER.info(f"Saved pointcloud. {filename=}")


def pointcloud_cut(pcd: o3d.geometry.PointCloud, cut_height: float, visualise_level: VisualiseLevel) -> np.ndarray:
    """Take 2D cut of pointcloud at specified height."""
    LOGGER.info("Taking 2D cut")

    points = np.asarray(pcd.points)
    y_val = points[:, 1]

    z_slice = points[(y_val < (cut_height + CUT_TOLERANCE)) & (y_val > cut_height)]
    z_slice_pcd = convert_to_pointcloud(z_slice)
    z_slice_pcd = algorithms.remove_pointcloud_outliers(z_slice_pcd, std_ratio=0.5, visualise_level=visualise_level)

    z_slice_points = np.asarray(z_slice_pcd.points)

    cut = np.delete(z_slice_points, 1, 1)  # remove y axis column
    cut = cut[:, [1, 0]]  # swap to local coordinates
    # cut[:, 0] = -cut[:, 0]  # assume leading edge further from wall

    if visualise_level >= VisualiseLevel.HIGH:
        plt.scatter(cut[:, 0], cut[:, 1], marker='.')
        plt.axis('equal')
        plt.show()

    return cut


def normalize_cut(
    cut: np.ndarray,
    visualise_level: VisualiseLevel,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Scale cut x-axis to between 0 and 1. Shift y-axis to mean 0."""
    LOGGER.info("Normalizing cut")

    zero_slice = cut[(cut[:, 1] < EDGE_IDENTIFICATION_TOL) & (cut[:, 1] > -EDGE_IDENTIFICATION_TOL)]

    leading_edge = zero_slice[zero_slice[:, 0] < EDGE_RELATIVE_MAX_POS]
    trailing_edge = zero_slice[zero_slice[:, 0] > (1 - EDGE_RELATIVE_MAX_POS)]

    leading_edge_pos = np.mean(leading_edge[:, 0])
    trailing_edge_pos = np.mean(trailing_edge[:, 0])

    blade_start = (leading_edge_pos, 0)
    blade_end = (trailing_edge_pos, 0)

    cut = (cut - leading_edge_pos) / (trailing_edge_pos - leading_edge_pos)
    cut[:, 1] = cut[:, 1] - np.mean(cut[:, 1])

    if visualise_level >= VisualiseLevel.HIGH:
        plt.scatter(cut[:, 0], cut[:, 1], marker='.')
        plt.axis('equal')
        plt.show()

    return cut, blade_start, blade_end


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)  # silence matplotlib debug messages

    parser = argparse.ArgumentParser()
    parser.add_argument('filename')

    args = parser.parse_args()

    pcd = open_pointcloud(args.filename)
    post_process(pcd, VisualiseLevel.OFF)
