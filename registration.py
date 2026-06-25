import ants
from pathlib import Path

def register_to_atlas(native_img, atlas_img, atlas_mask):
    """
    Performs a non-linear (SyN) registration of a native image to an atlas space.
    
    Args:
        native_img: ANTsImage representing the native scan (e.g. T1c).
        atlas_img: ANTsImage representing the fixed atlas template.
        atlas_mask: ANTsImage representing the brain mask of the atlas.
        
    Returns:
        tuple containing:
            - t1c_in_atlas_img: The native image registered to atlas space and skull-stripped.
            - registration: A dictionary containing the forward and inverse transforms.
    """
    registration = ants.registration(
        fixed=atlas_img, 
        moving=native_img, 
        type_of_transform='SyN',
        mask=atlas_mask
    )
    
    # Apply atlas mask for skull stripping in atlas space
    t1c_in_atlas_img = registration['warpedmovout'] * atlas_mask
    
    return t1c_in_atlas_img, registration

def transform_to_native(native_img, img_in_atlas, inv_transforms):
    """
    Transforms an image or mask from atlas space back to native space.
    
    Args:
        native_img: The original ANTsImage in native space (used as reference geometry).
        img_in_atlas: The ANTsImage to be transformed back.
        inv_transforms: List of paths to the inverse transforms.
        
    Returns:
        ANTsImage: The transformed image in native space.
    """
    # Build inversion list dynamically:
    # Affine transform (.mat) needs to be inverted (True)
    # Warp field (.nii.gz) is already inverse and should not be inverted (False)
    whichtoinvert = [True if str(t).endswith('.mat') else False for t in inv_transforms]
    
    native_space_img = ants.apply_transforms(
        fixed=native_img, 
        moving=img_in_atlas,
        transformlist=inv_transforms,
        whichtoinvert=whichtoinvert,
        interpolator='nearestNeighbor'
    )
    
    return native_space_img
