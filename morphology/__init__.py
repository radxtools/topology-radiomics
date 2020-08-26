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


def compute_curvature(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    FootNotes:
    [1] Surface Curvature Function conversion in python: https://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
    """
    # First Derivatives
    Xv, Xu = np.gradient(X)
    Yv, Yu = np.gradient(Y)
    Zv, Zu = np.gradient(Z)

    # Second Derivatives
    Xuv, Xuu = np.gradient(Xu)
    Yuv, Yuu = np.gradient(Yu)
    Zuv, Zuu = np.gradient(Zu)

    Xvv, Xuv = np.gradient(Xv)
    Yvv, Yuv = np.gradient(Yv)
    Zvv, Zuv = np.gradient(Zv)

    # Reshape to 1D vectors
    r, c = Xu.shape
    nrow = r*c  # total number of rows after reshaping
    Xu = Xu.reshape(nrow, 1)
    Yu = Yu.reshape(nrow, 1)
    Zu = Zu.reshape(nrow, 1)
    Xv = Xv.reshape(nrow, 1)
    Yv = Yv.reshape(nrow, 1)
    Zv = Zv.reshape(nrow, 1)
    Xuu = Xuu.reshape(nrow, 1)
    Yuu = Yuu.reshape(nrow, 1)
    Zuu = Zuu.reshape(nrow, 1)
    Xuv = Xuv.reshape(nrow, 1)
    Yuv = Yuv.reshape(nrow, 1)
    Zuv = Zuv.reshape(nrow, 1)
    Xvv = Xvv.reshape(nrow, 1)
    Yvv = Yvv.reshape(nrow, 1)
    Zvv = Zvv.reshape(nrow, 1)

    Xu = np.c_[Xu, Yu, Zu]
    Xv = np.c_[Xv, Yv, Zv]
    Xuu = np.c_[Xuu, Yuu, Zuu]
    Xuv = np.c_[Xuv, Yuv, Zuv]
    Xvv = np.c_[Xvv, Yvv, Zvv]

    # % First fundamental Coeffecients of the surface (E,F,G)
    # Einsum explained: https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428
    E = np.einsum('ij,ij->i', Xu, Xu)
    F = np.einsum('ij,ij->i', Xu, Xv)
    G = np.einsum('ij,ij->i', Xv, Xv)

    m = np.cross(Xu, Xv, axisa=1, axisb=1)
    p = np.sqrt(np.einsum('ij,ij->i', m, m))
    n = m/np.c_[p, p, p]

    # % Second fundamental Coeffecients of the surface (L,M,N)
    L = np.einsum('ij,ij->i', Xuu, n)
    M = np.einsum('ij,ij->i', Xuv, n)
    N = np.einsum('ij,ij->i', Xvv, n)

    # % Gaussian Curvature
    K = (L*N-M**2)/(E*G-L**2)
    K = K.reshape(r, c)

    # % Mean Curvature
    H = (E*N + G*L - 2*F*M)/(2*(E*G - F**2))
    H = H.reshape(r, c)

    # % Principle Curvatures
    Pmax = H + np.sqrt(H**2 - K)
    Pmin = H - np.sqrt(H**2 - K)

    curvature = Curvature(gaussian_curvature=K,
                          mean_curvature=H,
                          principal_curvature_max=Pmax,
                          principal_curvature_min=Pmin)
    return curvature

# todo: what does dim do? looks like its scalaing the meshgrid?


def compute_morphology_features(mri_voxels: np.ndarray, dim: np.ndarray = np.array([1, 1, 1])):
    n2, n1, n3 = mri_voxels.shape
    d1, d2, d3 = dim[0], dim[1], dim[2]
    xs = np.arange(n1) * d1
    ys = np.arange(n2) * d2
    zs = np.arange(n3) * d3

    # do we need this, as the python function cant use this data?
    X, Y, Z = np.meshgrid(xs, ys, zs)

    # todo: is this equivalent to smooth3 in ocatve?
    smoothed_mri_voxels = ndimage.gaussian_filter(mri_voxels, sigma=3)
    # todo: is this equivalent to isosurface in matlab?
    verts, faces, normals, values = measure.marching_cubes(smoothed_mri_voxels)
    _isosurface = Isosurface(verts, faces, normals, values)

    # cieling square for verticies??
    dim = 0
    l = len(verts)
    for n in range(l//2):
        if((n*n) > l):
            dim = n
            break

    v1 = np.zeros((dim*dim, 3))
    # todo - Verify this is correct with Rob Toth
    v1[0:l, :] = verts[0:l, :]

    XX = np.reshape(v1[:, 0], (dim, dim))
    YY = np.reshape(v1[:, 1], (dim, dim))
    ZZ = np.reshape(v1[:, 2], (dim, dim))

    _curvature = compute_curvature(XX, YY, ZZ)
    p_max = _curvature.principal_curvature_max
    p_min = _curvature.principal_curvature_min

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


def run(nii_path):
    """
    Wont have something like this in the final release (Used for testing)
    """
    img = nib.load(nii_path)
    mri_3d_voxels = img.get_fdata()
    mri_3d_voxels[mri_3d_voxels > 0] = 255
    response = compute_morphology_features(mri_3d_voxels)
    return response


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