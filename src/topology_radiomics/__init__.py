from topology_radiomics.morphology_features import compute_morphology_features
from topology_radiomics.config import MorphologyConfig
from topology_radiomics.sanitization import convert_volume_into_mask, convert_volume_into_multiple_masks, BinaryVoxelMask
from pkg_resources import resource_filename


def get_recurrence_sample_nii_path():
    nii_path = resource_filename(
        "topology_radiomics", "data/mask_recurrence.nii")
    return nii_path


def get_sample_nii_path():
    nii_path = resource_filename("topology_radiomics", "data/mask_PsP.nii")
    return nii_path


def get_sample_psuedo_progression_path():
    nii_path = resource_filename("topology_radiomics", "data/mask_PsP.nii")
    return nii_path
