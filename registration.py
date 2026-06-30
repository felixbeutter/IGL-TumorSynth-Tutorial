import ants

def register_to_atlas(native_img, atlas_img, atlas_mask):
    """
    Step 1: Non-linear (SyN) registration of native T1c image to atlas space.

    Args:
        native_img: ANTsImage representing the native scan (e.g. T1c).
        atlas_img: ANTsImage representing the fixed atlas template.
        atlas_mask: ANTsImage representing the brain mask of the atlas.

    Returns:
        tuple[ants.ANTsImage, dict]: (t1c_in_atlas_img, registration)
            t1c_in_atlas_img: The skull-stripped T1c image registered to atlas space.
            registration: Dictionary containing forward and inverse transform paths.
    """
    # Run the ANTs registration with Symmetric Normalization (SyN) deformable model.
    # We restrict registration using the fixed brain mask to ignore non-brain/skull regions.
    registration = ants.registration(
        fixed=atlas_img, 
        moving=native_img, 
        type_of_transform='SyN',
        mask=atlas_mask
    )
    
    # Multiply the warped output image by the template brain mask.
    # This skull-strips the patient's image in atlas space, removing surrounding tissues.
    t1c_in_atlas_img = registration['warpedmovout'] * atlas_mask
    
    return t1c_in_atlas_img, registration

def transform_to_native(native_img, img_in_atlas, inv_transforms):
    """
    Step 5: Transform registered image or mask from atlas space back to native space.

    Args:
        native_img: The original ANTsImage in native space (used as reference geometry).
        img_in_atlas: The ANTsImage to be transformed back.
        inv_transforms: List of paths to the inverse transforms.

    Returns:
        ants.ANTsImage: The transformed image in native space.
    """
    # Build inversion list dynamically:
    # 1. The linear affine transform (.mat) must be inverted (True) to map atlas -> native.
    # 2. The nonlinear warp field (.nii.gz) is already stored as an inverse warp from the
    #    moving-to-fixed registration, so it should not be inverted (False).
    whichtoinvert = [True if str(t).endswith('.mat') else False for t in inv_transforms]
    
    # Apply the transforms back to the original native image space.
    # We use nearest-neighbor interpolation to preserve the discrete integer values of masks/labels.
    native_space_img = ants.apply_transforms(
        fixed=native_img, 
        moving=img_in_atlas,
        transformlist=inv_transforms,
        whichtoinvert=whichtoinvert,
        interpolator='nearestNeighbor'
    )
    
    return native_space_img

def transform_to_atlas(atlas_img, moving_img, fwd_transforms):
    """
    Apply forward transformations to map a native image into the atlas space.

    Args:
        atlas_img: The fixed ANTsImage in atlas space (used as reference geometry).
        moving_img: The ANTsImage in native space to be transformed.
        fwd_transforms: List of paths to the forward transforms.

    Returns:
        ants.ANTsImage: The transformed image in atlas space.
    """
    # For mapping from native to atlas, we use the forward transforms as they are.
    atlas_space_img = ants.apply_transforms(
        fixed=atlas_img, 
        moving=moving_img,
        transformlist=fwd_transforms,
        interpolator='linear'
    )
    
    return atlas_space_img
