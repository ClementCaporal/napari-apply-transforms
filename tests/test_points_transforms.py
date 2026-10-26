import numpy as np

from napari_apply_transforms._points_transforms import (
    apply_affine_transform,
    apply_rotate_transform,
    apply_scale_transform,
    apply_shear_transform,
    apply_translate_transform,
)

points_coordinates_3D = np.array([[1, 2, 3], [4, 5, 6]])


def test_apply_affine_transform():
    affine_matrix = np.array(
        [[1, 0, 0, 1], [0, 2, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
    )
    transformed_points = apply_affine_transform(
        points_coordinates_3D, affine_matrix
    )
    expected_points = np.array([[2, 6, 6], [5, 12, 9]])

    assert np.allclose(transformed_points, expected_points)


def test_apply_translate_transform():
    translation_vector = np.array([1, 2, 3])
    transformed_points = apply_translate_transform(
        points_coordinates_3D, translation_vector
    )
    expected_points = np.array([[2, 4, 6], [5, 7, 9]])

    assert np.allclose(transformed_points, expected_points)


def test_apply_rotate_transform():
    angle = np.pi / 2  # 90 degrees
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    transformed_points = apply_rotate_transform(
        points_coordinates_3D, rotation_matrix
    )
    expected_points = np.array([[-2, 1, 3], [-5, 4, 6]])

    assert np.allclose(transformed_points, expected_points)


def test_apply_scale_transform():
    scale_factor = np.array([2, 3, 4])
    transformed_points = apply_scale_transform(
        points_coordinates_3D, scale_factor
    )
    expected_points = np.array([[2, 6, 12], [8, 15, 24]])

    assert np.allclose(transformed_points, expected_points)


def test_apply_shear_transform():
    shear = np.eye(3)
    shear[2, 0] = 0.5
    transformed_points = apply_shear_transform(points_coordinates_3D, shear)
    expected_points = np.array([[1, 2, 3.5], [4, 5, 8]])

    assert np.allclose(transformed_points, expected_points)
