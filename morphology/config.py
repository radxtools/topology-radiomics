from typing import Tuple
import enum


class MarchingCubesAlgorithm(enum.Enum):
    Lewiner = "lewiner"
    Lorensen = "lorensen"


class MorphologyConfig:
    """
    This class is used to provide configurations for the compute_morphology_features function 
    """
    voxel_spacing: Tuple[float, float, float] = (1., 1., 1.)
    gaussian_iterations: int = 1
    gaussian_sigma: float = 3.
    marching_cubes_algorithm: MarchingCubesAlgorithm = MarchingCubesAlgorithm.Lewiner
    marching_cubes_step_size: int = 1
