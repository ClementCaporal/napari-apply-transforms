"""
This module contains the function to apply transforms to layers in napari.
"""

from typing import TYPE_CHECKING

import numpy as np

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
) -> 'napari.types.LayerDataTuple':
    transformed_points = _apply_points_transforms(
        points,
        apply_affine=apply_affine,
        apply_rotate=apply_rotate,
        apply_scale=apply_scale,
        apply_shear=apply_shear,
        apply_translate=apply_translate,
    )

    # Start from the source layer's own metadata (size, colors, features,
    # units, axis labels, transforms, ...), then override what changed.
    _, meta, layer_type = points.as_layer_data_tuple()
    meta = dict(meta)
    ndim = points.ndim

    # A distinct name avoids overwriting the source layer.
    meta['name'] = f'{points.name} [transformed]'

    # Reset the transforms that are now baked into the coordinates, so that
    # napari does not apply them a second time. Transforms that were NOT
    # applied are kept from the source layer so the new layer behaves the same as the source layer.
    if apply_scale:
        meta['scale'] = np.ones(ndim)
    if apply_translate:
        meta['translate'] = np.zeros(ndim)
    if apply_rotate:
        meta['rotate'] = np.eye(ndim)
    if apply_shear:
        meta['shear'] = np.eye(ndim)
    if apply_affine:
        meta['affine'] = np.eye(ndim + 1)

    # Remove keys that are not valid in the layer data tuple, to avoid errors when
    # adding the new layer to the viewer.
    for key in ('ndim', 'property_choices'):
        meta.pop(key, None)

    return (transformed_points, meta, layer_type)
