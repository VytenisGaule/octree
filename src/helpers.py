import laspy
import numpy as np
from numpy import ndarray


def read_las_file(file_path: str) -> ndarray:
    """Read '.las' file return points"""
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    return points

def get_points_within_sphere(points: ndarray, center: tuple, radius: float) -> ndarray:
    """Get points within sphere, discard points outside sphere"""
    distances = np.linalg.norm(points - center, axis=1)
    return points[distances <= radius]

def get_sphere_volume(radius: float) -> float:
    """ Calculate sphere volume """
    return (4/3) * np.pi * (radius ** 3)

def get_cube_volume(side_length: float) -> float:
    """ Calculate cube volume """
    return side_length ** 3

def get_sphere_center(cube_min: float, cube_max: float) -> float:
    """ Calculate sphere center """
    return (cube_min + cube_max) / 2

def get_cube_size(points: ndarray)-> float:
    """ Calculate cube size """
    return np.max(points, axis=0) - np.min(points, axis=0)