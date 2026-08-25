from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ._line_profile import position_to_point

MEASUREMENT_VERSION = "line-profile-measurement-v3"


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
    cancel_check: Callable[[], bool] | None = None,
) -> LineMeasurement | None:
    """Measure bright stripe candidates along a centerline.

    The function is deterministic and has no Shape or Qt side effects. It uses
    local background and contrast on each normal, so a global image threshold
    cannot turn unrelated bright regions into a fixed-width result.
    """
    parameters = parameters or MeasurementParameters()
    gray = _as_gray(image)
    intensity_scale = _intensity_scale(image)
    total_length = _polyline_length(points)
    if total_length == 0:
        return LineMeasurement(MEASUREMENT_VERSION, ())
    count = max(2, int(math.ceil(total_length / parameters.sample_spacing)) + 1)
    samples_list: list[MeasurementSample] = []
    for index in range(count):
        if cancel_check is not None and cancel_check():
            return None
        samples_list.append(
            _measure_sample(
                gray=gray,
                points=points,
                position=index / (count - 1),
                parameters=parameters,
                intensity_scale=intensity_scale,
            )
        )
    samples = tuple(samples_list)
    return LineMeasurement(
        MEASUREMENT_VERSION, _recommend_neighbor_widths(samples=samples)
    )


def _measure_sample(
    *,
    gray: NDArray[np.float64],
    points: Sequence[Sequence[float]],
    position: float,
    parameters: MeasurementParameters,
    intensity_scale: float,
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
        parameters.search_radius + 0.25,
        0.5,
        dtype=np.float64,
    )
    coordinates = np.array(
        [
            [center[0] + offset * normal[0], center[1] + offset * normal[1]]
            for offset in offsets
        ]
    )
    values, valid = _sample_bilinear(gray, coordinates)
    if not valid.any():
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "outside_image"
        )

    # A global percentile is easily dominated by a glare patch or a second
    # line.  Estimate the background independently at both ends of the normal
    # and remove its linear trend before looking for the stripe.
    sample_values = _fill_invalid(values=values, valid=valid, offsets=offsets)
    outer_radius = max(2.0, parameters.search_radius * 0.65)
    left_outer = valid & (offsets <= -outer_radius)
    right_outer = valid & (offsets >= outer_radius)
    if left_outer.sum() < 2 or right_outer.sum() < 2:
        outer = valid
        left_background = right_background = float(np.percentile(values[outer], 35))
        left_reference = -parameters.search_radius
        right_reference = parameters.search_radius
        outer_residual = values[outer] - np.median(values[outer])
    else:
        left_background = float(np.percentile(values[left_outer], 35))
        right_background = float(np.percentile(values[right_outer], 35))
        left_reference = float(np.median(offsets[left_outer]))
        right_reference = float(np.median(offsets[right_outer]))
        outer = left_outer | right_outer
        outer_baseline = np.interp(
            offsets[outer],
            [left_reference, right_reference],
            [left_background, right_background],
        )
        outer_residual = values[outer] - outer_baseline
    baseline = np.interp(
        offsets,
        [left_reference, right_reference],
        [left_background, right_background],
    )
    residual = _smooth_profile(sample_values - baseline)
    outer_scale = 1.4826 * float(
        np.median(np.abs(outer_residual - np.median(outer_residual)))
    )
    # A small noise floor is intentional.  Without it, a 3-5 level intensity
    # change in a nearly flat 8-bit image is incorrectly treated as perfectly
    # visible merely because the local peak-to-peak range is also small.
    noise = max(2.0, outer_scale)

    # The annotation is a prior for where the line should be, not a promise
    # that its centre pixel is the brightest pixel.  Search a limited band so
    # a one-pixel drawing error or a dim centre does not truncate the stripe,
    # while a nearby glare streak cannot take over the measurement wholesale.
    candidate_radius = min(8.0, max(2.0, parameters.search_radius * 0.25))
    candidate = valid & (np.abs(offsets) <= candidate_radius)
    if not candidate.any():
        return MeasurementSample(
            position, parameters.min_width, 0.0, 0.0, "outside_image"
        )
    candidate_indices = np.flatnonzero(candidate)
    bright_index = int(candidate_indices[np.argmax(residual[candidate])])
    dark_index = int(candidate_indices[np.argmin(residual[candidate])])
    if residual[bright_index] >= -residual[dark_index]:
        center_index = bright_index
        polarity = 1.0
    else:
        center_index = dark_index
        polarity = -1.0
    signed_residual = polarity * residual
    signal = float(signed_residual[center_index])
    if signal <= 1.5 * noise:
        return MeasurementSample(
            position,
            parameters.min_width,
            0.0,
            0.0,
            "low_contrast" if signal <= noise else "low_visibility",
        )

    threshold = signal * parameters.contrast_factor
    evidence = valid & (signed_residual >= threshold)
    left = center_index
    while left > 0 and evidence[left - 1]:
        left -= 1
    right = center_index
    while right + 1 < len(evidence) and evidence[right + 1]:
        right += 1
    left_edge = (
        _threshold_crossing(
            offsets[left - 1],
            signed_residual[left - 1],
            offsets[left],
            signed_residual[left],
            threshold,
        )
        if left > 0
        else float(offsets[left])
    )
    right_edge = (
        _threshold_crossing(
            offsets[right],
            signed_residual[right],
            offsets[right + 1],
            signed_residual[right + 1],
            threshold,
        )
        if right + 1 < len(offsets)
        else float(offsets[right])
    )
    width = max(0.0, right_edge - left_edge)
    width = max(parameters.min_width, min(parameters.max_width, width))

    # Visibility combines local signal-to-noise and stripe/background contrast.
    # The separate display-intensity factor below prevents a dim bright stripe
    # in a flat dark crop from receiving a perfect score from a tiny denominator.
    signal_visibility = max(
        0.0,
        min(1.0, (signal - 1.5 * noise) / max(signal + 1.5 * noise, 1e-9)),
    )
    background_level = abs(float(baseline[center_index]))
    background_denominator = (
        background_level + 1.5 * noise
        if polarity < 0
        else signal + background_level + 1.5 * noise
    )
    background_contrast = max(
        0.0,
        min(
            1.0,
            signal / max(background_denominator, 1e-9),
        ),
    )
    # SNR alone gives the same score to a 35-on-2 stripe and a 235-on-202
    # stripe.  The former may be visible on a black background, while the
    # latter is nearly lost in a bright glare field.  Blend both measurements
    # before applying the display-intensity factor for bright stripes.
    visibility = 0.65 * signal_visibility + 0.35 * background_contrast
    if polarity > 0:
        # On the supplied 8-bit welding images, a line at intensity 30-40 on
        # a black background is barely visible even though its local contrast
        # is technically strong.  Keep local contrast as the primary signal,
        # but also account for the displayed intensity of a bright stripe.
        display_floor = 0.08 * intensity_scale
        display_ceiling = 0.86 * intensity_scale
        display_factor = max(
            0.0,
            min(
                1.0,
                (sample_values[center_index] - display_floor)
                / max(display_ceiling - display_floor, 1e-9),
            ),
        )
        visibility *= display_factor
    gradient = np.abs(np.gradient(signed_residual, offsets))
    edge_quality = min(
        1.0,
        float(np.mean((gradient[left], gradient[right])))
        / max(signal / max(width, 1.0), 1e-9),
    )
    confidence = max(0.0, min(1.0, 0.8 * visibility + 0.2 * edge_quality))
    reason = None if confidence >= 0.5 else "low_confidence"
    return MeasurementSample(position, width, visibility, confidence, reason)


def _as_gray(image: NDArray[np.generic]) -> NDArray[np.float64]:
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float64, copy=False)
    if array.ndim == 3 and array.shape[2] >= 3:
        return (
            0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
        ).astype(np.float64)
    raise ValueError("image must be a grayscale or RGB array")


def _intensity_scale(image: NDArray[np.generic]) -> float:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return float(np.iinfo(array.dtype).max)
    finite = array[np.isfinite(array)]
    if finite.size == 0 or float(np.max(finite)) <= 1.5:
        return 1.0
    return 255.0


def _recommend_neighbor_widths(
    *, samples: tuple[MeasurementSample, ...]
) -> tuple[MeasurementSample, ...]:
    """Fill uncertain widths from nearby reliable samples without raising confidence."""
    reliable = tuple(sample for sample in samples if sample.confidence >= 0.5)
    if not reliable:
        return samples
    result: list[MeasurementSample] = []
    for sample in samples:
        if sample.confidence >= 0.5:
            result.append(sample)
            continue
        left = next(
            (
                candidate
                for candidate in reversed(reliable)
                if candidate.position < sample.position
            ),
            None,
        )
        right = next(
            (
                candidate
                for candidate in reliable
                if candidate.position > sample.position
            ),
            None,
        )
        if left is not None and right is not None:
            ratio = (sample.position - left.position) / (right.position - left.position)
            width = left.width + ratio * (right.width - left.width)
        else:
            nearest = min(
                reliable,
                key=lambda candidate: abs(candidate.position - sample.position),
            )
            width = nearest.width
        result.append(dataclasses.replace(sample, width=width))
    return tuple(result)


def _sample_bilinear(
    image: NDArray[np.float64], coordinates: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    height, width = image.shape
    x_float = coordinates[:, 0]
    y_float = coordinates[:, 1]
    valid = (
        (x_float >= 0)
        & (x_float <= width - 1)
        & (y_float >= 0)
        & (y_float <= height - 1)
    )
    x = np.clip(np.floor(x_float).astype(int), 0, width - 1)
    y = np.clip(np.floor(y_float).astype(int), 0, height - 1)
    x1 = np.minimum(x + 1, width - 1)
    y1 = np.minimum(y + 1, height - 1)
    tx = x_float - x
    ty = y_float - y
    values = np.zeros(len(coordinates), dtype=np.float64)
    values[:] = (
        image[y, x] * (1 - tx) * (1 - ty)
        + image[y, x1] * tx * (1 - ty)
        + image[y1, x] * (1 - tx) * ty
        + image[y1, x1] * tx * ty
    )
    values[~valid] = 0.0
    return values, valid


def _fill_invalid(
    *,
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    offsets: NDArray[np.float64],
) -> NDArray[np.float64]:
    if valid.all():
        return values
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return values
    return np.interp(offsets, offsets[valid], values[valid])


def _smooth_profile(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(values) < 3:
        return values
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float64) / 4.0
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _threshold_crossing(
    left_offset: float,
    left_value: float,
    right_offset: float,
    right_value: float,
    threshold: float,
) -> float:
    denominator = right_value - left_value
    if denominator == 0:
        return (left_offset + right_offset) / 2.0
    fraction = (threshold - left_value) / denominator
    return left_offset + max(0.0, min(1.0, fraction)) * (right_offset - left_offset)


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(points, points[1:])
    )
