from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np
import skimage.restoration
import skimage.transform
from numpy.typing import NDArray

_MIN_DOWNSAMPLED_IMAGE_SIZE: Final[int] = 1
_IMAGE_DIMENSIONS: Final[int] = 3
_IMAGE_CHANNEL_DIMENSION: Final[int] = 2
_SUPPORTED_CHANNEL_COUNTS: Final[tuple[int, ...]] = (1, 3, 4)
_RGB_VALUE_MAX: Final[float] = 255.0
_TV_MAX_WEIGHT: Final[float] = 0.4


@dataclasses.dataclass(frozen=True)
class AiImageInput:
    image: NDArray[np.uint8]
    scale_x: float
    scale_y: float


def prepare_ai_image(
    *,
    image: NDArray[np.uint8],
    downsample_scale: float = 1.0,
    denoise_strength: float = 0.0,
) -> AiImageInput:
    """Prepare a model-only image while preserving the displayed source image."""
    if not 0.0 <= downsample_scale <= 1.0:
        raise ValueError(
            "downsample_scale must be between 0.0 and 1.0, "
            f"but got {downsample_scale!r}"
        )
    if not 0.0 <= denoise_strength <= 1.0:
        raise ValueError(
            "denoise_strength must be between 0.0 and 1.0, "
            f"but got {denoise_strength!r}"
        )
    if image.ndim != _IMAGE_DIMENSIONS or (
        image.shape[_IMAGE_CHANNEL_DIMENSION] not in _SUPPORTED_CHANNEL_COUNTS
    ):
        raise ValueError(
            "Expected an image with 1, 3, or 4 channels, "
            f"got {image.shape}"
        )

    original_height, original_width = image.shape[:2]
    if downsample_scale == 1.0:
        prepared = image
    else:
        prepared_height = max(
            _MIN_DOWNSAMPLED_IMAGE_SIZE,
            round(original_height * downsample_scale),
        )
        prepared_width = max(
            _MIN_DOWNSAMPLED_IMAGE_SIZE,
            round(original_width * downsample_scale),
        )
        prepared = skimage.transform.resize(
            image,
            output_shape=(prepared_height, prepared_width),
            order=1,
            mode="reflect",
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

    if denoise_strength != 0.0:
        # TV denoising suppresses isolated background variation while
        # preserving strong edges. The normalized UI value controls the TV
        # regularization weight; zero skips the filter entirely.
        denoised = skimage.restoration.denoise_tv_chambolle(
            prepared.astype(np.float32) / _RGB_VALUE_MAX,
            weight=denoise_strength * _TV_MAX_WEIGHT,
            channel_axis=-1,
        )
        prepared = (
            denoised * _RGB_VALUE_MAX
        ).clip(0, _RGB_VALUE_MAX).astype(np.uint8)

    return AiImageInput(
        image=prepared,
        scale_x=prepared.shape[1] / original_width,
        scale_y=prepared.shape[0] / original_height,
    )
