"""
Some utility functions that are shared across checks
"""
import numpy as np
from scipy.ndimage import find_objects


def remove_edge_labels(labeled_array: np.ndarray) -> np.ndarray:
    """ Removes labels from the input labeled_array that have at least
    one element that touches an edge. Operates in place.
    """
    # get all the unique patch ids from the edges of the labeled array
    top = labeled_array[0]
    bottom = labeled_array[labeled_array.shape[0] - 1]
    left = labeled_array[:, 0]
    right = labeled_array[:, labeled_array.shape[1] - 1]

    # we only want one array that contains the unique patch ids that
    # touch the edge of array
    all_edges = np.concatenate((top, bottom, left, right))
    all_edges_unique = np.unique(all_edges)
    # remove 0 as these pixels have data (and aren't holes)
    all_edges_unique = all_edges_unique[all_edges_unique != 0]

    # get the location of all patches
    object_slices = find_objects(labeled_array)
    for obj_slice in object_slices:
        # get the patch data
        patch = labeled_array[obj_slice]
        # now replace the only patch data that is a hole that touches an
        # edge. It's possible for a hole that touches an edge to suround
        # a hole that doesn't touch the edge. eg; the 2 below
        # [1 0 2 0 1 0 ]
        # [1 0 0 0 1 0 ]
        # [1 1 1 1 1 0 ]
        # [0 0 0 0 0 0 ]
        patch = np.where(
            np.isin(
                patch,
                all_edges_unique,
                assume_unique=False),
            0,
            patch
        )
        # now replace the data in the labeled array with the updated
        # patch
        labeled_array[obj_slice] = patch
    
    return labeled_array
