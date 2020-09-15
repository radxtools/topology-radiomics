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
                               The higher this value, the more smooth the lesion will become.
        | gaussian_sigma: Standard deviation for Gaussian kernel.
        | marching_cubes_algorithm: The marching cubes algorithm to use.
        | marching_cubes_step_size: The marching cubes step size to use.
        | clip_percent: The percents of data points to clip at the end of the distribution. 
                        Values in the outer percentile is clamped to the edge values, 
                Example: 
                    >> clip_percent = .2
                    >> x = list(range(0, 10))
                    [0,1,2,3,4,5,6,7,8,9]
                    >> clip(x)
                    [1,1,2,3,4,5,6,7,8,8]

                range: [0,1]
                .1 means clip [0,.05) and (0.95,100] in the distribution
                https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    voxel_spacing: Tuple[float, float, float]
    gaussian_iterations: int
    gaussian_sigma: float
    marching_cubes_algorithm: MarchingCubesAlgorithm
    marching_cubes_step_size: int
    clip_percent: float


    def __init__(self, 
                 voxel_spacing: Tuple[float, float, float] = (1., 1., 1.),
                 gaussian_iterations: int = 3,
                 gaussian_sigma: float = 3.,
                 marching_cubes_algorithm: MarchingCubesAlgorithm = MarchingCubesAlgorithm.Lewiner,
                 marching_cubes_step_size: int = 1,
                 clip_percent: float = .1
                        ):
        self.voxel_spacing = voxel_spacing
        self.gaussian_iterations = gaussian_iterations
        self.gaussian_sigma = gaussian_sigma
        self.marching_cubes_algorithm = marching_cubes_algorithm
        self.marching_cubes_step_size = marching_cubes_step_size
        self.clip_percent = clip_percent
        self._validate()

    def _validate(self):
        if not (0. <= self.clip_percent and self.clip_percent <= 1.):
            raise ValueError(f"clip_percent is not within range [0,1], given {self.clip_percent}")
