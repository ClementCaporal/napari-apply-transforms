"""
Bake a points layer's transforms into its coordinates, reusing napari's own
transform classes (napari.utils.transforms) instead of re-implementing them.
"""

import napari
import numpy as np
from napari.utils.transforms import CompositeAffine, TransformChain


def _apply_points_transforms(
    points_layer: 'napari.layers.Points',
    apply_affine: bool = True,
    apply_rotate: bool = True,
    apply_scale: bool = True,
    apply_shear: bool = True,
    apply_translate: bool = True,
) -> 'napari.types.PointsData':
    """
    Apply the layer's transforms to its data, in napari's own order:
    scale -> shear -> rotate -> translate (CompositeAffine), then layer.affine.

    Components that are switched off are replaced by their identity value.
    """
    ndim = points_layer.ndim
    data = np.array(points_layer.data, dtype=float, copy=True)

    # CompositeAffine builds rotate @ shear @ scale, then adds translate.
    # rotate/shear=None mean identity.
    composite = CompositeAffine(
        scale=points_layer.scale if apply_scale else np.ones(ndim),
        translate=points_layer.translate
        if apply_translate
        else np.zeros(ndim),
        rotate=points_layer.rotate if apply_rotate else None,
        shear=points_layer.shear if apply_shear else None,
        ndim=ndim,
    )

    chain = TransformChain([composite])
    if apply_affine:
        chain.append(points_layer.affine)

    return chain(data)
