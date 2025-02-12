import logging
import laspy
from laspy.lasdata import LasData
import numpy as np
from numpy import ndarray
from models import Octree
from visualization import visualize_octree
import os


logging.basicConfig(level=logging.INFO)

def main():
    """ main function """

    las_file: str = os.getenv('LAS_FILE', None)
    if not las_file:
        logging.error("LAS file not specified in environment variables")
        return
    
    try:
        las: LasData = laspy.read(las_file)
    except laspy.LaspyException as e:
        logging.error("Error reading LAS file: %s", e)
        return
    except FileNotFoundError as e:
        logging.error("LAS file not found: %s", e)
        return
    
    points: ndarray = np.vstack((las.x, las.y, las.z)).transpose()
    center: tuple = tuple(np.mean(points, axis=0))
    size: float = np.max(np.max(points, axis=0) - np.min(points, axis=0))
    octree: Octree = Octree(center, size)
    octree.insert_batch(points)
    is_visualized: bool = visualize_octree(octree)
    if is_visualized:
        logging.info("Octree visualized successfully")
    else:
        logging.error("Error visualizing octree")

if __name__ == "__main__":
    main()