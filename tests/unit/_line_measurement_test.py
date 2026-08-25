from __future__ import annotations

import numpy as np
import pytest

from labelme._line_measurement import MeasurementParameters
from labelme._line_measurement import MeasurementSample
from labelme._line_measurement import _recommend_neighbor_widths
from labelme._line_measurement import measure_line_profile


def test_measure_line_profile_detects_local_bright_stripe() -> None:
    image = np.full((48, 112), 20, dtype=np.uint8)
    image[17:24, 8:104] = 220

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [103.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    )

    assert result.measurement_version == "line-profile-measurement-v2"
    assert len(result.samples) >= 2
    middle = result.samples[len(result.samples) // 2]
    assert middle.width == pytest.approx(7.0, abs=0.5)
    assert middle.visibility > 0.9
    assert middle.confidence > 0.5


def test_measure_line_profile_reports_low_contrast_without_failure() -> None:
    image = np.full((32, 64), 100, dtype=np.uint8)

    result = measure_line_profile(
        image,
        [[4.0, 16.0], [60.0, 16.0]],
    )

    assert result.samples
    assert all(sample.confidence == 0.0 for sample in result.samples)
    assert all(sample.reason == "low_contrast" for sample in result.samples)


def test_measure_line_profile_recovers_width_when_centerline_is_off_center() -> None:
    image = np.full((64, 128), 20, dtype=np.uint8)
    image[12:28, 8:120] = 220

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [119.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=24, search_radius=16),
    )

    middle = result.samples[len(result.samples) // 2]
    assert middle.width > 14.0
    assert middle.visibility > 0.9


def test_measure_line_profile_does_not_call_a_dim_line_perfectly_visible() -> None:
    image = np.full((48, 112), 30, dtype=np.uint8)
    image[20:25, 8:104] = 35

    result = measure_line_profile(
        image,
        [[8.0, 22.0], [103.0, 22.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    )

    middle = result.samples[len(result.samples) // 2]
    assert middle.visibility < 0.3
    assert middle.confidence < 0.5


def test_measure_line_profile_supports_a_strong_dark_stripe() -> None:
    image = np.full((48, 112), 220, dtype=np.uint8)
    image[17:24, 8:104] = 20

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [103.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    )

    middle = result.samples[len(result.samples) // 2]
    assert middle.width == pytest.approx(7.0, abs=0.5)
    assert middle.visibility > 0.9


def test_measure_line_profile_validates_parameters_and_image() -> None:
    with pytest.raises(ValueError):
        MeasurementParameters(sample_spacing=0)
    with pytest.raises(ValueError, match="grayscale or RGB"):
        measure_line_profile(
            np.zeros((4, 4, 2), dtype=np.uint8),
            [[0.0, 0.0], [1.0, 1.0]],
        )


def test_low_confidence_width_uses_neighbor_recommendation_without_acceptance() -> None:
    samples = (
        MeasurementSample(0.0, 4.0, 1.0, 0.9),
        MeasurementSample(0.5, 1.0, 0.0, 0.1, "low_contrast"),
        MeasurementSample(1.0, 8.0, 1.0, 0.9),
    )

    recommended = _recommend_neighbor_widths(samples=samples)

    assert recommended[1].width == pytest.approx(6.0)
    assert recommended[1].confidence == 0.1
    assert recommended[1].reason == "low_contrast"


def test_measurement_can_cancel_between_samples() -> None:
    image = np.full((32, 128), 20, dtype=np.uint8)
    calls = 0

    def cancel_after_first_sample() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    result = measure_line_profile(
        image,
        [[4.0, 16.0], [120.0, 16.0]],
        cancel_check=cancel_after_first_sample,
    )

    assert result is None


@pytest.mark.parametrize(
    "points",
    [[], [[2.0, 2.0]], [[2.0, 2.0], [2.0, 2.0], [20.0, 2.0]]],
)
def test_measurement_handles_empty_short_and_repeated_centerlines(
    points: list[list[float]],
) -> None:
    result = measure_line_profile(
        np.zeros((32, 32), dtype=np.uint8),
        points,
        parameters=MeasurementParameters(search_radius=4.0),
    )

    assert result is not None
    if len(points) < 2 or points == [[2.0, 2.0], [2.0, 2.0]]:
        assert result.samples == ()
