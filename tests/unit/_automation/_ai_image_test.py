from __future__ import annotations

import numpy as np
import pytest

from labelme._automation._ai_image import prepare_ai_image


def test_default_preprocessing_keeps_original_image_unchanged() -> None:
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape((8, 10, 3))

    prepared = prepare_ai_image(image=image)

    assert prepared.image is image
    assert prepared.scale_x == 1.0
    assert prepared.scale_y == 1.0


def test_downsample_scale_changes_only_model_input_dimensions() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    prepared = prepare_ai_image(image=image, downsample_scale=0.3)

    assert prepared.image.shape == (3, 6, 3)
    assert prepared.scale_x == 0.3
    assert prepared.scale_y == 0.3
    assert image.shape == (10, 20, 3)


def test_zero_downsample_scale_keeps_a_valid_model_input() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    prepared = prepare_ai_image(image=image, downsample_scale=0.0)

    assert prepared.image.shape == (1, 1, 3)


def test_denoise_strength_reduces_background_noise_and_preserves_edges() -> None:
    image = np.zeros((9, 9, 3), dtype=np.uint8)
    image[:, 5:] = 255
    image[2, 2] = 255

    prepared = prepare_ai_image(image=image, denoise_strength=0.5)

    assert prepared.image.dtype == np.uint8
    assert prepared.image[2, 2, 0] < image[2, 2, 0]
    assert prepared.image[2, 4, 0] < 128
    assert prepared.image[2, 5, 0] > 128
    assert image[2, 2, 0] == 255


@pytest.mark.parametrize(
    ("downsample_scale", "denoise_strength"),
    [(-0.01, 0.0), (1.01, 0.0), (1.0, -0.01), (1.0, 1.01)],
)
def test_preprocessing_rejects_values_outside_range(
    *, downsample_scale: float, denoise_strength: float
) -> None:
    with pytest.raises(ValueError):
        prepare_ai_image(
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            downsample_scale=downsample_scale,
            denoise_strength=denoise_strength,
        )
