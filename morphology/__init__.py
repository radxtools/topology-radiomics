# %%
import numpy as np
from numpy import gradient
from collections import namedtuple
from typing import NamedTuple, Tuple
import scipy.ndimage as ndimage
from skimage import measure
import nibabel as nib
from scipy.stats import describe
import pyvista as pv

# https://web.njit.edu/~rlopes/Mod5.2.pdf
# https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html#arrays-dtypes-constructing


def compute_morphology_features1(mri_voxels: np.ndarray, spacing: Tuple[float, float, float] = (1., 1., 1.)):

    smoothed_mri_voxels = ndimage.gaussian_filter(mri_voxels, sigma=3)
    verts, faces, normals, values = measure.marching_cubes(
        smoothed_mri_voxels, spacing=spacing)
    faces_rows = faces.shape[0]
    poly_faces = np.column_stack(
        [3*np.ones((faces_rows, 1), dtype=np.int), faces])
    _isosurface = Isosurface(verts, faces, normals, values)
    surface = pv.PolyData(verts, poly_faces.flatten())

    mean_curvature = np.array(surface.curvature("Mean"))
    gaussian_curvature = np.array(surface.curvature("Gaussian"))
    p_min = np.array(surface.curvature("Minimum"))
    p_max = np.array(surface.curvature("Maximum"))

    _curvature = Curvature(
        gaussian_curvature=gaussian_curvature,
        mean_curvature=mean_curvature,
        principal_curvature_min=p_min,
        principal_curvature_max=p_max)

    _curvedness = .5 * np.sqrt(p_min ** 2 + p_max**2)
    _sharpness = (p_max - p_min)**2
    _shape_index = 2/np.pi * np.arctan((p_max + p_min) / (p_max - p_min))
    _total_curvature = np.abs(p_max) + np.abs(p_min)

    _surface_measures = SurfaceMeasures(
        curvedness=_curvedness, sharpness=_sharpness, shape_index=_shape_index, total_curvature=_total_curvature)

    return MorphologyFeatures(
        curvature=_curvature,
        surface_measures=_surface_measures,
        isosurface=_isosurface
    )


class Curvature(NamedTuple):
    gaussian_curvature: np.ndarray
    mean_curvature: np.ndarray
    principal_curvature_min: np.ndarray
    principal_curvature_max: np.ndarray

    def __repr__(self):
        return (
            f"Describe:\n"
            f"Gaussian curvature: {describe(self.gaussian_curvature.flatten(),nan_policy='omit')}\n"
            f"Mean Curvature: {describe(self.mean_curvature.flatten(),nan_policy='omit')}\n"
            f"Principal Curvature Min: {describe(self.principal_curvature_min.flatten(),nan_policy='omit')}\n"
            f"Principal Curvature Max: {describe(self.principal_curvature_max.flatten(),nan_policy='omit')}\n"

            f"Shapes:\n"
            f"Gaussian curvature: {self.gaussian_curvature.shape}\n"
            f"Mean Curvature: {self.mean_curvature.shape}\n"
            f"Principal Curvature Min: {self.principal_curvature_min.shape}\n"
            f"Principal Curvature Max: {self.principal_curvature_max.shape}")


class SurfaceMeasures(NamedTuple):
    curvedness: np.ndarray
    sharpness: np.ndarray
    shape_index: np.ndarray
    total_curvature: np.ndarray

    def __repr__(self):
        return (
            f"Describe:\n"
            f"Curvedness:{describe(self.curvedness.flatten(),nan_policy='omit')}\n"
            f"Sharpness:{describe(self.sharpness.flatten(),nan_policy='omit')}\n"
            f"Shape Index:{describe(self.shape_index.flatten(),nan_policy='omit')}\n"
            f"Total Curvature:{describe(self.total_curvature.flatten(),nan_policy='omit')}\n"

            f"Shapes:\n"
            f"Curvedness:{self.curvedness.shape}\n"
            f"Sharpness:{self.sharpness.shape}\n"
            f"Shape Index:{self.shape_index.shape}\n"
            f"Total Curvature:{self.total_curvature.shape}")


class Isosurface(NamedTuple):
    verts: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    values: np.ndarray


class MorphologyFeatures(NamedTuple):
    curvature: Curvature
    surface_measures: SurfaceMeasures
    isosurface: Isosurface

    def __repr__(self):
        return (f"Curvature:\n{self.curvature}\n"
                f"Surface measures:\n{self.surface_measures}")


def run1(nii_path):
    """
    Wont have something like this in the final release (Used for testing)
    """
    img = nib.load(nii_path)
    mri_3d_voxels = img.get_fdata()
    mri_3d_voxels[mri_3d_voxels > 0] = 255
    response = compute_morphology_features1(mri_3d_voxels)
    return response


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    run1(path)
