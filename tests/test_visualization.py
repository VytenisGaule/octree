import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import Octree
from visualization import plot_octree, plot_points, visualize_octree, plot_sphere


def test_plot_sphere():
    """ Test plot_sphere """
    fig: Figure = plt.figure()
    axes_obj = fig.add_subplot(111, projection='3d')
    center = (0, 0, 0)
    size = 1.0
    plot_sphere(axes_obj, center, size)
    
    # No data plotted
    assert len(axes_obj.collections) > 0
    plt.close(fig)
    
    
def test_plot_points():
    """ Test plot_points """
    fig: Figure = plt.figure()
    axes_obj = fig.add_subplot(111, projection='3d')

    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3]
    ])
    plot_points(axes_obj, points)
    # plot contains the expected data
    assert len(axes_obj.collections) > 0, "No data plotted"
    
    # coordinates of the plotted points
    scatter = axes_obj.collections[0]
    x_data, y_data, z_data = scatter._offsets3d
    expected_x = points[:, 0]
    expected_y = points[:, 1]
    expected_z = points[:, 2]
    assert np.allclose(x_data, expected_x)
    assert np.allclose(y_data, expected_y)
    assert np.allclose(z_data, expected_z)
    plt.close(fig)
    
    
def test_plot_octree():
    """ Test plot_octree """
    fig: Figure = plt.figure()
    axes_obj = fig.add_subplot(111, projection='3d')
    center: tuple = (0, 0, 0)
    size: float = 1.0
    octree = Octree(center, size)
    points: np.ndarray = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3]
    ])
    octree.insert_batch(points)
    plot_octree(axes_obj, octree)
    assert len(axes_obj.collections) > 0
    plt.close(fig)

def test_visualize_octree():
    """ Test visualize_octree """
    
    # Test correct data
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3]
    ])
    octree.insert_batch(points)
    is_octree_visualized: bool = visualize_octree(octree)
    assert True is is_octree_visualized
    
    # Test corrupted data
    octree.points[0] = ["a", "b", "c"] 
    is_octree_visualized = visualize_octree(octree)
    assert is_octree_visualized is False
