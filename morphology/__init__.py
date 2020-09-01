import numpy as np
from typing import NamedTuple, Tuple
import scipy.ndimage as ndimage
from skimage import measure
import pyvista as pv
from pyvista import PolyData


class Curvature:
    """
    Datam to hold Curvature features
    """

    def computed_gaussian_curvature(self) -> np.ndarray:
        """Computes the gaussian curvature for each vertex
        returns:
            ndarray: Shape: (N,)
        """
        return self.gaussian_curvature

    def computed_mean_curvature(self) -> np.ndarray:
        """Computes the mean curvature for each vertex
        returns:
            ndarray: Shape: (N,)
        """
        return self.mean_curvature

    def computed_principal_curvature_min(self) -> np.ndarray:
        """Computes the minimum principal curvature for each vertex
        returns:
            ndarray: Shape: (N,)
        """
        return self.principal_curvature_min

    def computed_principal_curvature_max(self) -> np.ndarray:
        """Computes the maximum principal curvature for each vertex
        returns:
            ndarray: Shape: (N,)
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
            self.computed_gaussian_curvature(), "Gaussian Curvuture")
        mean = SummaryRow(self.computed_mean_curvature(), "Mean Curvuture")
        pmin = SummaryRow(
            self.computed_principal_curvature_min(), "Principal Curvature Min")
        pmax = SummaryRow(
            self.computed_principal_curvature_max(), "Principal Curvature Max")
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

    def computed_curvedness(self) -> np.ndarray:
        """Cache of curvedness for each vertex

        Formulas is given as follows:
            P1 = principal curvature minimum
            P2 = principal curvature maximum
            Curvedness       C = 5 * sqrt(P1^2 + P2^2)

            The curvedness (C) captures flat regions in the surface with low
            curvedness values and regions of sharp curvature having high
            curvedness

        returns:
            ndarray: Shape: (N,)
        """
        return self.curvedness

    def computed_sharpness(self) -> np.ndarray:
        """Cache of sharpness for each vertex

        Formulas is given as follows:
            P1 = principal curvature minimum
            P2 = principal curvature maximum
            Sharpness        S = (P2 - P1)^2

         The sharpness degree (S) measures the sharpness of the curvature by relating the
         mean curvature H to the actual surface

         returns:
            ndarray: Shape: (N,)
        """
        return self.sharpness

    def computed_shape_index(self) -> np.ndarray:
        """Cache of shape index for each vertex

        Formulas is given as follows:
            P1 = principal curvature minimum
            P2 = principal curvature maximum
            Shape Index     SI = 2/pi * arctan((P2 + P1) / (P1 - P2))
            TODO: what to do when P1 - P2 = 0?

        The shape index (SI) is a number ranging from 1 to 1 that provides a continuous 
        gradation between shapes. It is sensitive to subtle changes in surface shape, 
        particularly in regions where total curvature is very low. For instance, hollow
        structures have a shape index of <0, and inflections and bumps have a shape index of > 0

        returns:
            ndarray: Shape: (N,)
        """
        return self.shape_index

    def computed_total_curvature(self) -> np.ndarray:
        """Cache of total curvature for each vertex
        Formulas is given as follows:
            P1 = principal curvature minimum
            P2 = principal curvature maximum
            Total Curvature  K = abs(P1) + abs(P2)

        The total curvature (KT) was computed to obtain the absolute value of the total curvature
        at each surface voxel

        returns:
            ndarray: Shape: (N,)
        """
        return self.total_curvature

    def __str__(self):
        header = SummaryRow.print_header()
        curvedness_summary = SummaryRow(
            self.computed_curvedness(), "Curvedness")
        sharpness_summary = SummaryRow(self.computed_sharpness(), "Sharpness")
        shape_index_summary = SummaryRow(
            self.computed_curvedness(), "Shape Index")
        total_curvature_summary = SummaryRow(
            self.computed_curvedness(), "Total Curvature")
        return "\n".join(map(str, [header, curvedness_summary, sharpness_summary, shape_index_summary, total_curvature_summary]))


class Isosurface(NamedTuple):
    """
    This class is useful when you need draw the surface.

    Attributes:
        verts: list of the vertex (x,y,z), Shape: (N x 3)
        faces: list with three items which contains the vertex index
               ex:  
               verts = [ [0,0,0], [2,2,2], [0,1,0]]
               faces = [[0 1 2]]
        normals : TODO
        values : TODO

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
        self.curvature = curvature
        self.surface_measures = surface_measures
        self.surface_mesh = surface_mesh
        self.isosurface = isosurface

    def get_curvature(self) -> Curvature:
        """
        Getter for Curvature
        """
        return self.curvature

    def get_isosurface(self) -> Isosurface:
        """
        Getter for IsoSurface

        This class contains information to create a mesh or surface plot for visalizations
        """
        return self.isosurface

    def get_surface_measures(self) -> SurfaceMeasures:
        """
        Getter for Surface Measures
        """
        return self.surface_measures

    def __str__(self):
        return (f"Curvature:\n{self.curvature}\n"
                f"Surface measures:\n{self.surface_measures}")


def compute_morphology_features(mri_mask_voxels: np.ndarray, spacing: Tuple[float, float, float] = (1., 1., 1.)) -> MorphologyFeatures:
    """
    This function will compute the surface measures as published in (TODO: paper link)

    High Level overview of the algorithm:
        Gaussian Filter -> Marching Cubes -> PolyData Surface -> Results

    Typical usage example:
        img = nib.load(nii_path)
        mri_3d_voxels: numpy.ndarray = img.get_fdata()
        # contains only 0 or 255 (255 is the maximum pixel value)
        mri_3d_voxels[mri_3d_voxels > 0] = 255
        features_data = compute_morphology_features(mri_3d_voxels)
    """
    smoothed_mri_mask_voxels = ndimage.gaussian_filter(
        mri_mask_voxels, sigma=3)
    verts, faces, normals, values = measure.marching_cubes(
        smoothed_mri_mask_voxels, spacing=spacing)
    faces_rows = faces.shape[0]
    poly_faces = np.column_stack(
        [3*np.ones((faces_rows, 1), dtype=np.int), faces])
    _isosurface = Isosurface(verts, faces, normals, values)
    surface = pv.PolyData(verts, poly_faces.flatten())

    _curvature = _compute_curvature(surface)
    _surface_measures = _compute_surface_measures(_curvature)

    return MorphologyFeatures(
        curvature=_curvature,
        surface_measures=_surface_measures,
        surface_mesh=surface,
        isosurface=_isosurface
    )


def _compute_curvature(surface: pv.PolyData) -> Curvature:
    "Private helper function to retrieve curvature based on Polydata computed curvature values for each vertex"
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


def _compute_surface_measures(curvature: Curvature) -> SurfaceMeasures:
    """Private helper function to compute surface measures as published in (TODO: paper link)

        Formulas are given as follows:
            P1 = principal curvature minimum
            P2 = principal curvature maximum

            Curvedness       C = 5 * sqrt(P1^2 + P2^2)
            Sharpness        S = (P2 - P1)^2
            Shape Index     SI = 2/pi * arctan((P2 + P1) / (P1 - P2))
            Total Curvature  K = abs(P1) + abs(P2)
    """
    p_min = curvature.computed_principal_curvature_min()
    p_max = curvature.computed_principal_curvature_max()

    _curvedness = .5 * np.sqrt(p_min ** 2 + p_max**2)
    _sharpness = (p_max - p_min)**2
    _shape_index = 2/np.pi * np.arctan((p_max + p_min) / (p_max - p_min))
    _total_curvature = np.abs(p_max) + np.abs(p_min)

    _surface_measures = SurfaceMeasures(
        curvedness=_curvedness, sharpness=_sharpness, shape_index=_shape_index, total_curvature=_total_curvature)
    return _surface_measures


class SummaryRow():
    """
    Helper class to sumarize results for ndarrays
    """

    _HEADER = [
        "row_header",
        "shape",
        "min",
        "max",
        "mean",
        "std"
    ]

    _FORMAT = {
        "row_header": "{:25s}",
        "shape": "{}",
        "min": "{:+.4e}",
        "max": "{:+.4e}",
        "mean": "{:+.4e}",
        "std": "{:+.4e}"
    }

    def __init__(self, arr: np.ndarray, row_header: str = ""):
        self.row_header = row_header
        self.shape = arr.shape
        self.min = arr.min()
        self.max = arr.max()
        self.mean = arr.mean()
        self.std = arr.std()

    @staticmethod
    def print_header():
        header = "|".join(SummaryRow._HEADER)
        return header

    def __str__(self):
        values = [
            self._FORMAT["row_header"].format(self.row_header),
            self._FORMAT["shape"].format(self.shape),
            self._FORMAT["min"].format(self.min),
            self._FORMAT["max"].format(self.max),
            self._FORMAT["mean"].format(self.mean),
            self._FORMAT["std"].format(self.std)
        ]

        formated_values_repr = "|".join(values)
        return formated_values_repr


if __name__ == "__main__":
    try:
        import sys
        import nibabel as nib
        from pkg_resources import resource_filename

        nii_path = resource_filename("morphology", "data/Lesion.nii")
        # nii_path = sys.argv[1]
        img = nib.load(nii_path)
        mri_3d_voxels = img.get_fdata()
        mri_3d_voxels[mri_3d_voxels > 0] = 255
        features_data = compute_morphology_features(mri_3d_voxels)
        print(features_data)
    except:
        print("nibabel not installed. Can't run __main__")
