"""
This module contains the functions to apply transforms to points layers in napari.
"""

import napari
import numpy as np


def apply_rotate_transform(points, rotation_matrix):
    """
    Apply a rotation transformation to the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform in the shape (N, D), where N is the number of
        points and D is the dimensionality.
    rotation_matrix : np.ndarray
        The rotation matrix in the shape (D, D).

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    return points @ rotation_matrix.T


def apply_scale_transform(points, scale_factors):
    """
    Apply a scaling transformation to the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform in the shape (N, D).
    scale_factors : np.ndarray
        The scaling factors for each dimension in the shape (D,).

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    return points * scale_factors


def apply_shear_transform(points, shear_matrix):
    """
    Apply a shearing transformation to the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform in the shape (N, D).
    shear_matrix : np.ndarray
        The shearing matrix in the shape (D, D).

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    return points @ shear_matrix.T


def apply_translate_transform(points, translation_vector):
    """
    Apply a translation transformation to the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform in the shape (N, D).
    translation_vector : np.ndarray
        The translation vector in the shape (D,).

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    return points + translation_vector


def apply_affine_transform(points, affine_matrix):
    """
    Apply an affine transformation to the points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform in the shape (N, D).
    affine_matrix : np.ndarray
        The affine transformation matrix in the shape (D+1, D+1).

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    num_points = points.shape[0]

    # Convert points to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((num_points, 1))])

    # Apply the affine transformation
    transformed = homogeneous_points @ affine_matrix.T

    # Convert back to Cartesian coordinates (only divide if w != 1 / != 0)
    w = transformed[:, -1][:, np.newaxis]
    if not np.allclose(w, 1.0):
        w = np.where(np.isclose(w, 0.0), 1.0, w)
        return transformed[:, :-1] / w
    return transformed[:, :-1]


def _shear_to_matrix(shear, ndim):
    """Return a (D, D) shear matrix, whatever the napari version stores."""
    shear = np.asarray(shear, dtype=float)
    if shear.ndim == 2:
        return shear
    # Older napari: upper-triangular values stored as a flat vector
    matrix = np.eye(ndim)
    matrix[np.triu_indices(ndim, 1)] = shear
    return matrix


def _apply_points_transforms(
    points_layer: 'napari.layers.Points',
    apply_affine: 'bool' = True,
    apply_rotate: 'bool' = True,
    apply_scale: 'bool' = True,
    apply_shear: 'bool' = True,
    apply_translate: 'bool' = True,
) -> 'napari.types.PointsData':
    """
    Bake a points layer's transforms into its coordinates.

    Transforms are applied in the same order napari uses internally:
    scale -> shear -> rotate -> translate -> affine.
    """
    # Copy (as float) to avoid modifying the original data
    transformed_points = np.array(points_layer.data, dtype=float, copy=True)
    ndim = transformed_points.shape[1]

    if apply_scale:
        transformed_points = apply_scale_transform(
            transformed_points, np.asarray(points_layer.scale, dtype=float)
        )

    if apply_shear:
        transformed_points = apply_shear_transform(
            transformed_points, _shear_to_matrix(points_layer.shear, ndim)
        )

    if apply_rotate:
        transformed_points = apply_rotate_transform(
            transformed_points, np.asarray(points_layer.rotate, dtype=float)
        )

    if apply_translate:
        transformed_points = apply_translate_transform(
            transformed_points, np.asarray(points_layer.translate, dtype=float)
        )

    if apply_affine:
        transformed_points = apply_affine_transform(
            transformed_points,
            np.asarray(points_layer.affine.affine_matrix, dtype=float),
        )

    return transformed_points
