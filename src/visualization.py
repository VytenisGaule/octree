from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models import Octree
import logging
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
load_dotenv()

def plot_sphere(axes_obj: Axes3D, center: tuple, size: float):
    """Plot a sphere"""
    longitude = np.linspace(0, 2 * np.pi, 10)
    latitude = np.linspace(0, np.pi, 10)
    x = center[0] + (size / 2) * np.outer(np.cos(longitude), np.sin(latitude))
    y = center[1] + (size / 2) * np.outer(np.sin(longitude), np.sin(latitude))
    z = center[2] + (size / 2) * np.outer(np.ones(np.size(longitude)), np.cos(latitude))
    axes_obj.plot_wireframe(x, y, z, color="g", alpha=0.1)

def plot_points(axes_obj: Axes3D, points: np.ndarray, max_points_per_leaf: int = None):
    """Plot points within a sphere"""
    if points.size > 0:
        if max_points_per_leaf is not None and points.shape[0] > max_points_per_leaf:
            points = points[:max_points_per_leaf]
        axes_obj.scatter(points[:, 0], points[:, 1], points[:, 2], color='r', s=1)

def plot_octree(axes_obj: Axes3D, octree: Octree, max_points_per_leaf: int = None):
    """Plot octree"""
    try:
        if octree.is_leaf:
            plot_sphere(axes_obj, octree.center, octree.size)
            plot_points(axes_obj, np.array(octree.points), max_points_per_leaf)
        else:
            for child in octree.children:
                plot_octree(axes_obj, child, max_points_per_leaf)
    except Exception as e:
        logging.error("Error plotting octree: %s", e)
        raise
            
def visualize_octree(octree: Octree) -> bool:
    """Visualize octree and return True if successful, False otherwise"""
    try:
        DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
        fig: Figure = plt.figure()
        axes_obj: Axes3D = fig.add_subplot(111, projection='3d')
        if DEBUG:
            plot_octree(axes_obj, octree, max_points_per_leaf=10)
        else:
            plot_octree(axes_obj, octree)
        plt.savefig('plot.png')
        return True
    except Exception as e:
        logging.error("Error visualizing octree: %s", e)
        return False