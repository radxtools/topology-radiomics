from bric_radiomics.morphology_features import compute_morphology_features
from bric_radiomics.config import MorphologyConfig
from bric_radiomics.sanitization import convert_volume_into_mask, convert_volume_into_multiple_masks, BinaryVoxelMask
from pkg_resources import resource_filename

def get_sample_nii_path():
    nii_path = resource_filename("bric_radiomics", "data/mask_recurrence.nii")
    return nii_path