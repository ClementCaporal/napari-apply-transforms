import numpy as np
import pytest
from napari.layers import Points

from napari_apply_transforms._points_transforms import (
    _apply_points_transforms,
)

points_coordinates_3D = np.array([[1, 2, 3], [4, 5, 6]])

SCALE = np.array([2, 3, 4])
TRANSLATE = np.array([1, 2, 3])
ANGLE = np.pi / 2  # 90 degrees around z
ROTATE = np.array(
    [
        [np.cos(ANGLE), -np.sin(ANGLE), 0],
        [np.sin(ANGLE), np.cos(ANGLE), 0],
        [0, 0, 1],
    ]
)
SHEAR = np.eye(3)
SHEAR[0, 2] = 0.5  # x' = x + 0.5 * z
AFFINE = np.array([[1, 0, 0, 1], [0, 2, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

ALL_OFF = {
    'apply_affine': False,
    'apply_rotate': False,
    'apply_scale': False,
    'apply_shear': False,
    'apply_translate': False,
}


def only(**flags):
    """All flags off, except the ones given."""
    return {**ALL_OFF, **flags}


@pytest.fixture
def layer_all():
    """A layer with every transform set, so that each test can check that
    the switched-off components are really ignored."""
    return Points(
        points_coordinates_3D.copy(),
        scale=SCALE,
        translate=TRANSLATE,
        rotate=ROTATE,
        shear=SHEAR,
        affine=AFFINE,
    )


def test_no_flags_returns_data_unchanged(layer_all):
    result = _apply_points_transforms(layer_all, **ALL_OFF)
    np.testing.assert_allclose(result, points_coordinates_3D)


def test_apply_scale_only(layer_all):
    result = _apply_points_transforms(layer_all, **only(apply_scale=True))
    expected = np.array([[2, 6, 12], [8, 15, 24]])
    np.testing.assert_allclose(result, expected)


def test_apply_translate_only(layer_all):
    result = _apply_points_transforms(layer_all, **only(apply_translate=True))
    expected = np.array([[2, 4, 6], [5, 7, 9]])
    np.testing.assert_allclose(result, expected)


def test_apply_rotate_only(layer_all):
    result = _apply_points_transforms(layer_all, **only(apply_rotate=True))
    expected = np.array([[-2, 1, 3], [-5, 4, 6]])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_apply_shear_only(layer_all):
    result = _apply_points_transforms(layer_all, **only(apply_shear=True))
    expected = np.array([[2.5, 2, 3], [7, 5, 6]])
    np.testing.assert_allclose(result, expected)


def test_apply_affine_only(layer_all):
    result = _apply_points_transforms(layer_all, **only(apply_affine=True))
    expected = np.array([[2, 6, 6], [5, 12, 9]])
    np.testing.assert_allclose(result, expected)


def test_all_flags_follow_napari_order(layer_all):
    """scale -> shear -> rotate -> translate, then layer.affine."""
    result = _apply_points_transforms(layer_all)

    linear = ROTATE @ SHEAR @ np.diag(SCALE)
    composite = points_coordinates_3D @ linear.T + TRANSLATE
    homogeneous = np.hstack([composite, np.ones((len(composite), 1))])
    expected = (homogeneous @ AFFINE.T)[:, :-1]

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_all_flags_match_napari_data_to_world(layer_all):
    if not hasattr(layer_all, 'data_to_world'):
        pytest.skip('Layer.data_to_world not available in this napari version')
    result = _apply_points_transforms(layer_all)
    # data_to_world takes ONE position (length ndim), not an (N, D) array
    expected = np.array([layer_all.data_to_world(p) for p in layer_all.data])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_translation_is_not_rotated():
    """Order matters: rotate-then-translate (napari) != translate-then-rotate."""
    layer = Points(
        points_coordinates_3D.copy(), rotate=ROTATE, translate=TRANSLATE
    )
    result = _apply_points_transforms(layer)

    rotate_then_translate = points_coordinates_3D @ ROTATE.T + TRANSLATE
    translate_then_rotate = (points_coordinates_3D + TRANSLATE) @ ROTATE.T

    np.testing.assert_allclose(result, rotate_then_translate, atol=1e-12)
    assert not np.allclose(result, translate_then_rotate)


def test_layer_and_input_data_not_modified(layer_all):
    original = layer_all.data.copy()
    _apply_points_transforms(layer_all)
    np.testing.assert_array_equal(layer_all.data, original)


def test_output_is_float_and_keeps_shape():
    layer = Points(points_coordinates_3D.copy(), scale=[0.5, 0.5, 0.5])
    result = _apply_points_transforms(layer)
    assert result.shape == points_coordinates_3D.shape
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, points_coordinates_3D * 0.5)
