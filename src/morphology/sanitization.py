import numpy as np
from abc import ABC
from typing import List
import logging

logger = logging.getLogger("morphology.sanitization")

_3D_PRINCIPAL_AXIS_COUNT = 3


def _validate_3dimensions(arr: np.ndarray):
    if len(arr.shape) != _3D_PRINCIPAL_AXIS_COUNT:
        raise InvalidDimensionsException(arr)


class InvalidDimensionsException(Exception):
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.msg = f"Shape of array is {arr.shape}. Expected (m,n,p)"


def _validate_binary_volume(arr: np.ndarray):
    if len(np.unique(arr)) > 2:
        raise VolumeNotBinaryException(arr)


class VolumeNotBinaryException(Exception):
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.msg = f"Volume has more than two values: {np.unique(arr)}. Expected all elemnts to be {{ 0,1 }}"


class BinaryVoxelMask:
    mri_voxel_mask: np.ndarray = None

    def __init__(self, mri_volume: np.ndarray):
        _validate_3dimensions(mri_volume)
        _validate_binary_volume(mri_volume)
        self.mri_voxel_mask = mri_volume


class OutOfBoundsException(Exception):
    def __init__(self, expected_bound, actual_bound):
        self.msg = f"Out of bounds exception. Expected {expected_bound} but Actual {actual_bound}"


def convert_volume_into_multiple_masks(mri_volume: np.ndarray, unions: List[int]) -> List[BinaryVoxelMask]:
    """
    Utility function to create multiple masks. See `convert_volume_into_mask`
    """
    masks = []
    for u in unions:
        masks.append(convert_volume_into_mask(mri_volume, u))
    return masks


def convert_volume_into_mask(mri_volume: np.ndarray, union: int) -> BinaryVoxelMask:
    """
    This will create a binary mask based on the labels in the voxel volume and will combine multiple labels into a single value based on union condition.
    The labels selected represented by the union number will be 1, all other values will be 0.
    The minimum value in the mri_volume is considered as empty space
    union: 
        labels = sorted . unique . mri_volume
        Imagine if labels for the volume is [0,1,2,4]
        There are 4 possible values for the label.
        The number of subsets for this list is 2^4:
        Each subset can represent which numbers you would like to union together.

        The below table has a mapping of all possible union combinations:

        ================
        4 2 1 0 | union
        ================
        0 0 0 0 =   0
        0 0 0 1 =   1
        0 0 1 0 =   2
        0 0 1 1 =   3
        0 1 0 0 =   4
        0 1 0 1 =   5
        0 1 1 0 =   6
        0 1 1 1 =   7
        1 0 0 0 =   8
        1 0 0 1 =   9
        1 0 1 0 =  10
        1 0 1 1 =  11
        1 1 0 0 =  12
        1 1 0 1 =  13
        1 1 1 0 =  14
        1 1 1 1 =  15

        Ex:

        Example 1:
            labels 1 and 2 should be the same label.

            This means you would like to union(1,2)
                                                    4 2 1 0
            This is the same as the subset {1,2} = [0 1 1 0] = 6


        Example 2:
            labels 1,2 and 4 should be the same label.

            This means you would like to union(1,2,4)
                                                      4 2 1 0
            This is the same as the subset {1,2,4} = [1 1 1 0] = 14
    """
    labels = sorted(list(np.unique(mri_volume)))
    logging.debug(f"Found Labels: {labels}")

    positions = '{0:b}'.format(abs(union))
    if  len(positions) < len(labels):
        padding = len(labels) - len(positions)
        # reversing positions to index into labels
        # adding padding to make it iterate every element and set unused labels to 0
        positions = (list(positions)[::-1]) + (['0'] * padding)

    if len(positions) > len(labels):
        raise OutOfBoundsException(actual_bound=len(
            positions), expected_bound=len(labels))

    mask = mri_volume.copy()

    for i, flag in enumerate(positions):
        label = labels[i]
        if flag == '1':
            logging.debug(f"Setting mask label {label} to 1")
            mask[mask == label] = 1
        else:
            logging.debug(f"Setting mask label {label} to 0")
            mask[mask == label] = 0
    return BinaryVoxelMask(mask)
