"""
This module contains the function to apply transforms layers in napari.
"""

from typing import TYPE_CHECKING

from ._points_transforms import _apply_points_transforms

if TYPE_CHECKING:
    import napari


def apply_points_transforms(
    points: 'napari.layers.Points',
    apply_affine: 'bool' = True,
    apply_rotate: 'bool' = True,
    apply_scale: 'bool' = True,
    apply_shear: 'bool' = True,
    apply_translate: 'bool' = True,
) -> 'napari.types.PointsData':
    transformed_points = _apply_points_transforms(
        points,
        apply_affine=apply_affine,
        apply_rotate=apply_rotate,
        apply_scale=apply_scale,
        apply_shear=apply_shear,
        apply_translate=apply_translate,
    )

    return transformed_points  # return the transformed points data
