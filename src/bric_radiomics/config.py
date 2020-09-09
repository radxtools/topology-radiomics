from typing import Tuple
import enum


class MarchingCubesAlgorithm(enum.Enum):
    Lewiner = "lewiner"
    Lorensen = "lorensen"


class MorphologyConfig:
    """
    This class is used to provide configurations for the compute_morphology_features function. You can consider these as hyperparameters.

    Attributes:
        | voxel_spacing: The spacing between each pixel.
        | gaussian_iterations: The number of iterations to smooth the image.
        | gaussian_sigma: Standard deviation for Gaussian kernel.
        | marching_cubes_algorithm: The marching cubes algorithm to use.
        | marching_cubes_step_size: The marching cubes step size to use.

    """
    voxel_spacing: Tuple[float, float, float] = (1., 1., 1.)
    gaussian_iterations: int = 1
    gaussian_sigma: float = 3.
    marching_cubes_algorithm: MarchingCubesAlgorithm = MarchingCubesAlgorithm.Lewiner
    marching_cubes_step_size: int = 1
