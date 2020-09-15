from bric_radiomics.sanitization import BinaryVoxelMask
from bric_radiomics.reporting import SummaryRow
from bric_radiomics.config import MorphologyConfig, MarchingCubesAlgorithm
import logging
from pyvista import PolyData
import pyvista as pv
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from typing import NamedTuple, Tuple
import numpy as np
from pandas import DataFrame

logger = logging.getLogger("bric_radiomics.morphology_features")


class Curvature:
    """
    Datam to hold Curvature features
    """

    @property
    def computed_gaussian_curvature(self) -> np.ndarray:
        """Computes the gaussian curvature for each vertex

        Returns:
            ndarray with Shape: (N,)
        """
        return self.gaussian_curvature

    @property
    def computed_mean_curvature(self) -> np.ndarray:
        """Computes the mean curvature for each vertex

        Returns:
            ndarray with Shape: (N,)
        """
        return self.mean_curvature

    @property
    def computed_principal_curvature_min(self) -> np.ndarray:
        """Computes the minimum principal curvature for each vertex

        Returns:
            ndarray with Shape: (N,)
        """
        return self.principal_curvature_min

    @property
    def computed_principal_curvature_max(self) -> np.ndarray:
        """Computes the maximum principal curvature for each vertex

        Returns:
            ndarray with Shape: (N,)
        """
        return self.principal_curvature_max

    def __init__(self,
                 gaussian_curvature: np.ndarray,
                 mean_curvature: np.ndarray,
                 principal_curvature_min: np.ndarray,
                 principal_curvature_max: np.ndarray):
        self.gaussian_curvature = gaussian_curvature
        self.mean_curvature = mean_curvature
        self.principal_curvature_max = principal_curvature_min
        self.principal_curvature_min = principal_curvature_max

    def __str__(self):
        header = SummaryRow.print_header()
        gauss = SummaryRow(
            self.computed_gaussian_curvature, "Gaussian Curvuture")
        mean = SummaryRow(self.computed_mean_curvature, "Mean Curvuture")
        pmin = SummaryRow(
            self.computed_principal_curvature_min, "Principal Curvature Min")
        pmax = SummaryRow(
            self.computed_principal_curvature_max, "Principal Curvature Max")
        return "\n".join(map(str, [header, gauss, mean, pmin, pmax]))


class SurfaceMeasures:
    """
    Datam to hold Surface Measures
    """

    def __init__(self,
                 curvedness: np.ndarray,
                 sharpness: np.ndarray,
                 shape_index: np.ndarray,
                 total_curvature: np.ndarray):
        self.curvedness = curvedness
        self.sharpness = sharpness
        self.shape_index = shape_index
        self.total_curvature = total_curvature

    @property
    def computed_curvedness(self) -> np.ndarray:
        """Cache of curvedness for each vertex

        Formulas is given as follows:

        .. math:: P1 = Principal Curvature Minimum

        .. math:: P2 = Principal Curvature Maximum

        .. math:: Curvedness :: C = 5 * sqrt(P1^2 + P2^2)

        The curvedness (C) captures flat regions in the surface with low
        curvedness values and regions of sharp curvature having high
        curvedness

        Returns:
            ndarray with Shape: (N,)
        """
        return self.curvedness

    @property
    def computed_sharpness(self) -> np.ndarray:
        """Cache of sharpness for each vertex

        Formulas is given as follows:

        .. math:: P1 = Principal Curvature Minimum

        .. math:: P2 = Principal Curvature Maximum

        .. math:: Sharpness :: S = (P2 - P1)^2

        The sharpness degree (S) measures the sharpness of the curvature by relating the
        mean curvature H to the actual surface

         Returns:
            ndarray with Shape: (N,)
        """
        return self.sharpness

    @property
    def computed_shape_index(self) -> np.ndarray:
        """Cache of shape index for each vertex

        Formulas is given as follows:

        .. math:: P1 = Principal Curvature Minimum

        .. math:: P2 = Principal Curvature Maximum

        .. math:: Shape Index:: SI = 2/pi * arctan((P2 + P1) / (P1 - P2)) 
        where to avoid divison by zero when P1 - P2 we add epsilon(10^-6)

        The shape index (SI) is a number ranging from -1 to 1 that provides a continuous 
        gradation between shapes. It is sensitive to subtle changes in surface shape, 
        particularly in regions where total curvature is very low. For instance, hollow
        structures have a shape index of <0, and inflections and bumps have a shape index of > 0

        Returns:
            ndarray with Shape: (N,)
        """
        return self.shape_index

    @property
    def computed_total_curvature(self) -> np.ndarray:
        """Cache of total curvature for each vertex

        Formulas is given as follows:

        .. math:: P1 = Principal Curvature Minimum

        .. math:: P2 = Principal Curvature Maximum

        .. math:: Total Curvature  K = abs(P1) + abs(P2)


        The total curvature (KT) was computed to obtain the absolute value of the total curvature
        at each surface voxel

        Returns:
            ndarray with Shape: (N,)
        """
        return self.total_curvature

    def __str__(self):
        header = SummaryRow.print_header()
        curvedness_summary = SummaryRow(
            self.computed_curvedness, "Curvedness")
        sharpness_summary = SummaryRow(self.computed_sharpness, "Sharpness")
        shape_index_summary = SummaryRow(
            self.shape_index, "Shape Index")
        total_curvature_summary = SummaryRow(
            self.total_curvature, "Total Curvature")
        return "\n".join(map(str, [header, curvedness_summary, sharpness_summary, shape_index_summary, total_curvature_summary]))


class Isosurface(NamedTuple):
    """This class is useful when you need draw the surface.

    This is a cache of the output from marching cubes algorithm.

    Attributes:

        verts: list of the vertex (x,y,z), Shape: (N x 3)

        faces: list with three items which contains the vertex index
                Example:  
                    >>> verts = [[0,0,0], [2,2,2], [0,1,0]]
                    >>> faces = [[0 1 2]]

        normals: The normal direction at each vertex, as calculated from the data. Shape: (N, 3)

        values: Gives a measure for the maximum value of the data in the local region near each vertex. This can be used by visualization tools to apply a colormap to the mesh. Shape: (N, )

    """
    verts: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    values: np.ndarray


class MorphologyFeatures:
    """
    Datam to hold Morphology Features
    """

    def __init__(self,
                 curvature: Curvature,
                 surface_measures: SurfaceMeasures,
                 surface_mesh: PolyData,
                 isosurface: Isosurface
                 ):
        self._curvature = curvature
        self._surface_measures = surface_measures
        self._surface_mesh = surface_mesh
        self._isosurface = isosurface

    def to_DF(self) -> DataFrame:
        """
        Returns:
            A useful dataframe that can be used to view the data
        """
        df = DataFrame({
            "shape_index": self.surface_measures.computed_shape_index,
            "curvedness": self.surface_measures.computed_curvedness,
            "sharpness": self.surface_measures.computed_sharpness,
            "total_curvature": self.surface_measures.computed_total_curvature,
            "x": self.isosurface.verts[:, 0],
            "y": self.isosurface.verts[:, 1],
            "z": self.isosurface.verts[:, 2],
            "face_idx": np.arange(self.surface_measures.computed_shape_index.shape[0])
        })
        return df

    @property
    def curvature(self) -> Curvature:
        """Getter for Curvature

        Returns:
            A Curvature object
        """
        return self._curvature

    @property
    def isosurface(self) -> Isosurface:
        """Getter for IsoSurface
        This class contains information to create a mesh or surface plot for visalizations

        Returns:
            The Isosurface created by marching cubes
        """
        return self._isosurface

    @property
    def surface_measures(self) -> SurfaceMeasures:
        """Getter for Surface Measures

        Returns:
            The Surface Measures
        """
        return self._surface_measures

    def __str__(self):
        return (f"Curvature:\n{self.curvature}\n"
                f"Surface measures:\n{self.surface_measures}")


def compute_morphology_features(mri_mask_voxels: BinaryVoxelMask,
                                config: MorphologyConfig = MorphologyConfig()) -> MorphologyFeatures:
    """This function will compute the surface measures as published in (TODO: paper link)

    High Level overview of the algorithm:
        Gaussian Filter -> Marching Cubes -> PolyData Surface -> Results

    Args:
        mri_mask_voxels: The mask of the voxel. Expected values for each element is {0,1}
        config: Configurations used for computing the morphology features

    Example:
        >>> import sys
        >>> import nibabel as nib
        >>> from pkg_resources import resource_filename
        >>> import logging
        >>> import bric_radiomics as br
        >>> FORMAT = '%(asctime)-15s %(levelname)s %(funcName)s  %(message)s'
        >>> logging.basicConfig(format=FORMAT, level=logging.DEBUG)       
        >>> nii_path = resource_filename("morphology", "data/mask_recurrence.nii")
        >>> img = nib.load(nii_path)
        >>> mri_3d_voxels = img.get_fdata().copy()
        >>> sanitized_voxels = br.convert_volume_into_mask(mri_3d_voxels,merge_labels=2)
        >>> features_data = compute_morphology_features(sanitized_voxels)

    Returns:
        MorphologyFeatures: an object with cached values

    """
    mask = mri_mask_voxels.mri_voxel_mask
    logger.debug(f"mask type is {mask.dtype}")
    logger.info(
        f"Starting Smoothing (iterations={config.gaussian_iterations}, sigma={config.gaussian_sigma})")
    logger.debug(
        f"Iteration 1: smoothing using sigma: {config.gaussian_sigma}")
    smoothed_mri_mask_voxels = mask
    logger.debug(f"smoothed_mri_mask type is {smoothed_mri_mask_voxels.dtype}")

    for i in range(config.gaussian_iterations):
        logger.debug(
            f"Iteration {i+1}: smoothing using sigma: {config.gaussian_sigma}")
        smoothed_mri_mask_voxels = gaussian_filter(
            smoothed_mri_mask_voxels, sigma=config.gaussian_sigma)
        logger.debug(
            f"smoothed_mri_mask type is {smoothed_mri_mask_voxels.dtype}")

    logger.info(
        f"Coverting volume into triangles using marching cubes (spacing={config.voxel_spacing},method={config.marching_cubes_algorithm},step_size={config.marching_cubes_step_size})")
    verts, faces, normals, values = marching_cubes(
        volume=smoothed_mri_mask_voxels,
        spacing=config.voxel_spacing,
        method=config.marching_cubes_algorithm.value,
        step_size=config.marching_cubes_step_size,
        allow_degenerate=False)

    faces_rows = faces.shape[0]
    poly_faces = np.column_stack(
        [3*np.ones((faces_rows, 1), dtype=np.int), faces])

    logger.info(f"Converting triangles into Mesh")
    _isosurface = Isosurface(verts, faces, normals, values)
    surface = pv.PolyData(verts, poly_faces.flatten())

    logger.info(f"Computing Curvature")
    _curvature = _compute_curvature(surface)
    logger.debug(f"Curvature:\n {_curvature}")

    logger.info(f"Computing Surface Measures")
    _surface_measures = _compute_surface_measures(_curvature, config.clip_percent)
    logger.debug(f"Surface Measures:\n {_surface_measures}")

    return MorphologyFeatures(
        curvature=_curvature,
        surface_measures=_surface_measures,
        surface_mesh=surface,
        isosurface=_isosurface
    )


def _compute_curvature(surface: pv.PolyData) -> Curvature:
    """Private helper function to retrieve curvature based on Polydata computed curvature values for each vertex
    Args:
        surface: the mesh created by the faces and vertices by running marching cubes

    Returns:
        A cached Curvature object for each vertex
    """
    mean_curvature = np.array(surface.curvature("Mean"))
    gaussian_curvature = np.array(surface.curvature("Gaussian"))
    p_min = np.array(surface.curvature("Minimum"))
    p_max = np.array(surface.curvature("Maximum"))

    _curvature = Curvature(
        gaussian_curvature=gaussian_curvature,
        mean_curvature=mean_curvature,
        principal_curvature_min=p_min,
        principal_curvature_max=p_max)
    return _curvature


def _clip(arr: np.ndarray, clip_percent: float, in_place=True) -> None:
    """Private helper function to clip extreme values.

    Clipping will set values over the population over the threshold to the value of the interval edge
    More information about clipping can be found here: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    """
    left_percentile = clip_percent / 2
    right_percentile = 1. - left_percentile
    l = np.quantile(arr, left_percentile, interpolation='nearest')
    r = np.quantile(arr, right_percentile, interpolation='nearest')
    out = None
    if in_place:
        out = arr
    results = np.clip(arr, a_min=l, a_max=r, out=out)
    return results
    


def _compute_surface_measures(curvature: Curvature, clip_percent: float) -> SurfaceMeasures:
    """Private helper function to compute surface measures as published in (TODO: paper link)

        Formulas are given as follows:

        .. math:: P1 = Principal Curvature Minimum

        .. math:: P2 = Principal Curvature Maximum

        .. math:: Curvedness :: C = 5 * sqrt(P1^2 + P2^2)

        .. math:: Sharpness :: S = (P2 - P1)^2

        .. math:: Shape Index :: SI = 2/pi * arctan((P2 + P1) / (P1 - P2))

        .. math:: Total Curvature :: K = abs(P1) + abs(P2)

        Args:
            curvature: The curvature measures

        Returns:
            A cached SurfaceMeasures object for each vertex
    """
    p_min = curvature.computed_principal_curvature_min
    p_max = curvature.computed_principal_curvature_max

    _curvedness = .5 * np.sqrt(p_min ** 2 + p_max**2)
    logger.debug(f"Clipping cuvature using {clip_percent}")
    _clip(_curvedness, clip_percent)

    _sharpness = (p_max - p_min)**2
    logger.debug(f"Clipping sharpness using {clip_percent}")
    _clip(_sharpness, clip_percent)
    """
    # division by zero if p_max is p_mix.
    # to avoid this. add tiny error term.
    # This should only be the case if either p_max or p_min is zero meaning the there is no curvature
    """
    diff = p_max - p_min
    diff[diff == 0] = 1e-6
    _shape_index = 2/np.pi * np.arctan((p_max + p_min) / diff)
    logger.debug(f"Clipping shape_index using {clip_percent}")
    _clip(_shape_index, clip_percent)

    _total_curvature = np.abs(p_max) + np.abs(p_min)
    logger.debug(f"Clipping total_curvature using {clip_percent}")
    _clip(_total_curvature, clip_percent)

    _surface_measures = SurfaceMeasures(
        curvedness=_curvedness,
        sharpness=_sharpness,
        shape_index=_shape_index,
        total_curvature=_total_curvature)
    return _surface_measures


if __name__ == "__main__":

    import sys
    import nibabel as nib
    from pkg_resources import resource_filename
    import logging
    import bric_radiomics as br

    FORMAT = '%(asctime)-15s %(levelname)s %(funcName)s  %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)

    nii_path = resource_filename("bric_radiomics", "data/Lesion.nii")
    nii_path = resource_filename("bric_radiomics", "data/mask_PsP.nii")
    nii_path = resource_filename("bric_radiomics", "data/mask_recurrence.nii")

    img = nib.load(nii_path)
    mri_3d_voxels = img.get_fdata().copy()
    sanitized_voxels = br.convert_volume_into_mask(
        mri_3d_voxels, merge_labels=[1, 2])
    features_data = compute_morphology_features(sanitized_voxels)
    print(features_data)
