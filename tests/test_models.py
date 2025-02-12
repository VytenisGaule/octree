import pytest
import sys
import os
import numpy as np

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models import Octree

def test_insert_point_within_sphere():
    """ Test inserting a point within the sphere """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    point = np.array([0.1, 0.1, 0.1])
    octree.insert_point(point)
    assert point.tolist() in [p.tolist() for p in octree.points]

def test_insert_point_outside_sphere():
    """ Test inserting a point outside the sphere """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    point = np.array([2.0, 2.0, 2.0])
    octree.insert_point(point)
    assert point.tolist() not in [p.tolist() for p in octree.points]

def test_insert_batch_valid_points():
    """ Test batch inserting valid points """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    points = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3]
    ])
    octree.insert_batch(points)
    # 2 points within sphere
    assert len(octree.points) == 2

def test_insert_batch_invalid_points():
    """ Test batch inserting invalid points """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    points = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3]
    ])
    with pytest.raises(ValueError):
        octree.insert_batch(points)

def test_is_within_sphere():
    """ Test checking if a point is within the sphere """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    point_within = np.array([0.1, 0.1, 0.1])
    point_outside = np.array([2.0, 2.0, 2.0])
    assert octree.is_within_sphere(point_within)
    assert not octree.is_within_sphere(point_outside)

def test_subdivide():
    """ Test subdividing the octree """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    octree.subdivide()
    assert len(octree.children) == 8

def test_is_leaf():
    """ Test checking if the node is a leaf node """
    center = (0, 0, 0)
    size = 1.0
    octree = Octree(center, size)
    assert octree.is_leaf
    octree.subdivide()
    assert not octree.is_leaf