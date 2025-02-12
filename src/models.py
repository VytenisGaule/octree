import numpy as np
from numpy import ndarray
import logging

logging.basicConfig(level=logging.INFO)

class Octree:
    """
    Octree data structure
    """
    def __init__(self, center: tuple, size: float):
        self.center: tuple = center
        self.size: float = size
        self.children: list = []
        self.points: list = []

    def insert_point(self, point: ndarray):
        """Insert a point into node or subdivide current node"""
        if not self.is_within_sphere(point):
            # discard points that are not within the sphere
            return
        
        if len(self.points) < 8 and not self.children:
            # Add the point to the current node if there is space
            self.points.append(point)
        else:
            if not self.children:
                # Subdivide the current node if leaf node exists
                self.subdivide()
            for child in self.children:
                # recursion - insert the point into the child node
                child.insert_point(point)
                
    def insert_batch(self, points: ndarray):
        """Batch insert points into the Octree"""
        try:
            for point in points:
                self.insert_point(point)
        except Exception as e:
            logging.error("Error inserting points: %s", e)
            raise ValueError("Invalid data provided")

    def is_within_sphere(self, point: ndarray) -> bool:
        """Check if the point is within the sphere"""
        distance: float = np.linalg.norm(np.array(point) - np.array(self.center))
        radius: float = self.size / 2
        return distance <= radius

    def subdivide(self):
        """Subdivide node into 8"""
        half_size: float = self.size / 2
        quarter_size: float = self.size / 4
        offsets: list = [
            (-quarter_size, -quarter_size, -quarter_size),
            (-quarter_size, -quarter_size, quarter_size),
            (-quarter_size, quarter_size, -quarter_size),
            (-quarter_size, quarter_size, quarter_size),
            (quarter_size, -quarter_size, -quarter_size),
            (quarter_size, -quarter_size, quarter_size),
            (quarter_size, quarter_size, -quarter_size),
            (quarter_size, quarter_size, quarter_size),
        ]
        for offset in offsets:
            child_center = tuple(np.array(self.center) + np.array(offset))
            self.children.append(Octree(child_center, half_size))
    
    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node"""
        return not self.children