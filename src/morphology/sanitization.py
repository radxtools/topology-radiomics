import numpy as np
from abc import ABC

_3D_PRINCIPAL_AXIS_COUNT = 3


def validate_3dimensions(arr: np.ndarray):
    if len(arr.shape) != _3D_PRINCIPAL_AXIS_COUNT:
        raise InvalidDimensionsException(arr)


class InvalidDimensionsException(Exception):
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.msg = f"Shape of array is {arr.shape}. Expected (m,n,p)"


def validate_binary_volume(arr: np.ndarray):
    if len(np.unique(arr)) > 2:
        raise VolumeNotBinaryException(arr)


class VolumeNotBinaryException(Exception):
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.msg = f"Volume has more than two values: {np.unique(arr)}. Expected all elemnts to be {{ 0,1 }}"


class SanitizedVolume(ABC):
    mri_volume : np.ndarray = None

class BinaryVolume(SanitizedVolume):
    def __init__(self, mri_volume: np.ndarray):
        validate_3dimensions(mri_volume)
        validate_binary_volume(mri_volume)
        self.mri_volume = mri_volume


def convert_volume_into_binary(mri_volume: np.ndarray, subset_number: int, subset: List[int] ):
    n = list(np.unique(mri_volume))