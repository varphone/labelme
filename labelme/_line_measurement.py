from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._line_profile import position_to_point

MEASUREMENT_VERSION = "line-profile-measurement-v1"


@dataclasses.dataclass(frozen=True)
class MeasurementParameters:
    sample_spacing: float = 8.0
    search_radius: float = 32.0
    min_width: float = 1.0
    max_width: float = 256.0
    contrast_factor: float = 0.35

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_spacing) or self.sample_spacing <= 0:
            raise ValueError("sample_spacing must be positive and finite")
        if not math.isfinite(self.search_radius) or self.search_radius <= 0:
            raise ValueError("search_radius must be positive and finite")
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("width bounds must be positive and ordered")
        if not 0 <= self.contrast_factor <= 1:
            raise ValueError("contrast_factor must be within [0, 1]")


@dataclasses.dataclass(frozen=True)
class MeasurementSample:
    position: float
    width: float
    visibility: float
    confidence: float
    reason: str | None = None


@dataclasses.dataclass(frozen=True)
class LineMeasurement:
    measurement_version: str
    samples: tuple[MeasurementSample, ...]


def measure_line_profile(
    image: NDArray[np.generic],
    points: Sequence[Sequence[float]],
    *,
    parameters: MeasurementParameters | None = None,
) -> LineMeasurement:
    """Measure bright stripe candidates along a centerline.

    The function is deterministic and has no Shape or Qt side effects. It uses
    local background and contrast on each normal, so a global image threshold
    cannot turn unrelated bright regions into a fixed-width result.
    """
    parameters = parameters or MeasurementParameters()
    gray = _as_gray(image)
    total_length = _polyline_length(points)
    if total_length == 0:
        return LineMeasurement(MEASUREMENT_VERSION, ())
    count = max(2, int(math.ceil(total_length / parameters.sample_spacing)) + 1)
    samples = tuple(
        _measure_sample(
            gray=gray,
            points=points,
            position=index / (count - 1),
            parameters=parameters,
        )
        for index in range(count)
    )
    return LineMeasurement(MEASUREMENT_VERSION, samples)


def _measure_sample(
    *,
    gray: NDArray[np.float64],
    points: Sequence[Sequence[float]],
    position: float,
    parameters: MeasurementParameters,
) -> MeasurementSample:
    center = position_to_point(points, position)
    before = position_to_point(points, max(0.0, position - 1e-4))
    after = position_to_point(points, min(1.0, position + 1e-4))
    dx, dy = after[0] - before[0], after[1] - before[1]
    tangent_length = math.hypot(dx, dy)
    if tangent_length == 0:
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "zero_length_tangent"
        )
    normal = (-dy / tangent_length, dx / tangent_length)
    offsets = np.arange(
        -parameters.search_radius,
        parameters.search_radius + 0.5,
        1.0,
        dtype=np.float64,
    )
    coordinates = np.array(
        [
            [center[0] + offset * normal[0], center[1] + offset * normal[1]]
            for offset in offsets
        ]
    )
    values, valid = _sample_nearest(gray, coordinates)
    if not valid.any():
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "outside_image"
        )
    valid_values = values[valid]
    background = float(np.percentile(valid_values, 20))
    center_index = int(np.argmin(np.abs(offsets)))
    center_value = float(values[center_index]) if valid[center_index] else background
    contrast = center_value - background
    if contrast <= 0:
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "low_contrast"
        )
    threshold = background + contrast * parameters.contrast_factor
    bright = valid & (values >= threshold)
    if not bright[center_index]:
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "no_center_evidence"
        )
    left = center_index
    while left > 0 and bright[left - 1]:
        left -= 1
    right = center_index
    while right + 1 < len(bright) and bright[right + 1]:
        right += 1
    width = float(offsets[right] - offsets[left] + 1.0)
    width = max(parameters.min_width, min(parameters.max_width, width))
    left_strength = float(values[center_index] - values[left])
    right_strength = float(values[center_index] - values[right])
    symmetry = 1.0 - abs(left_strength - right_strength) / max(contrast, 1e-9)
    visibility = max(0.0, min(1.0, contrast / max(np.ptp(valid_values), 1e-9)))
    confidence = max(0.0, min(1.0, 0.5 * visibility + 0.5 * symmetry))
    reason = None if confidence >= 0.5 else "low_confidence"
    return MeasurementSample(position, width, visibility, confidence, reason)


def _as_gray(image: NDArray[np.generic]) -> NDArray[np.float64]:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        return (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        ).astype(np.float64)
    raise ValueError("image must be a grayscale or RGB array")


def _sample_nearest(
    image: NDArray[np.float64], coordinates: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    height, width = image.shape
    x = np.rint(coordinates[:, 0]).astype(int)
    y = np.rint(coordinates[:, 1]).astype(int)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    values = np.zeros(len(coordinates), dtype=np.float64)
    values[valid] = image[y[valid], x[valid]]
    return values, valid


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )
