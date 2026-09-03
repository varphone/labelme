from __future__ import annotations

import numpy as np
import pytest

from labelme._line_measurement import MeasurementParameters
from labelme._line_measurement import MeasurementSample
from labelme._line_measurement import _recommend_neighbor_widths
from labelme._line_measurement import _sample_positions
from labelme._line_measurement import measure_line_profile


def test_measure_line_profile_detects_local_bright_stripe() -> None:
    image = np.full((48, 112), 20, dtype=np.uint8)
    image[17:24, 8:104] = 220

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [103.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    )

    assert result.measurement_version == "line-profile-measurement-v3"
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


def test_measure_line_profile_ignores_one_sided_glare_when_finding_width() -> None:
    image = np.full((100, 100), 20, dtype=np.uint8)
    image[17:24, 8:92] = 220
    image[24:40, 8:92] = 255

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [91.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=100, search_radius=20),
    )

    assert result.samples[0].width == pytest.approx(7.0, abs=1.0)


def test_measure_line_profile_ignores_one_sided_glare_for_dark_line() -> None:
    image = np.full((100, 100), 220, dtype=np.uint8)
    image[17:24, 8:92] = 20
    image[24:40, 8:92] = 255

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [91.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=100, search_radius=20),
    )

    assert result.samples[0].width == pytest.approx(7.0, abs=1.0)


def test_measure_line_profile_corrects_a_diffuse_asymmetric_glare_edge() -> None:
    image = np.full((100, 100), 20, dtype=np.uint8)
    image[17:24, 8:92] = 220
    image[24:34, 8:92] = np.linspace(220, 255, 10, dtype=np.uint8)[:, None]

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [91.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=100, search_radius=20),
    )

    sample = result.samples[0]
    assert sample.width == pytest.approx(7.0, abs=1.0)
    assert sample.confidence < 0.5
    assert sample.reason == "asymmetric_edges"


def test_measure_line_profile_merges_a_short_dark_notch_in_the_laser_envelope() -> None:
    image = np.full((100, 100), 20, dtype=np.uint8)
    image[17:20, 8:92] = 220
    image[20:23, 8:92] = 20
    image[23:27, 8:92] = 220

    result = measure_line_profile(
        image,
        [[8.0, 22.0], [91.0, 22.0]],
        parameters=MeasurementParameters(sample_spacing=100, search_radius=20),
    )

    assert result.samples[0].width == pytest.approx(10.0, abs=1.0)


def test_measure_line_profile_stabilizes_a_localized_glare_width_run() -> None:
    image = np.full((100, 120), 20, dtype=np.uint8)
    image[17:24, 8:112] = 220
    image[24:34, 35:75] = np.linspace(220, 255, 10, dtype=np.uint8)[:, None]

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [111.0, 20.0]],
        parameters=MeasurementParameters(sample_spacing=16, search_radius=20),
    )

    assert max(sample.width for sample in result.samples) < 9.0
    assert any(sample.reason == "asymmetric_edges" for sample in result.samples)


def test_measurement_samples_every_polyline_turn() -> None:
    image = np.full((80, 80), 100, dtype=np.uint8)

    result = measure_line_profile(
        image,
        [[8.0, 20.0], [48.0, 20.0], [48.0, 60.0]],
        parameters=MeasurementParameters(sample_spacing=100, search_radius=8),
    )

    assert [sample.position for sample in result.samples] == pytest.approx(
        [0.0, 0.5, 1.0]
    )


def test_sample_positions_deduplicate_repeated_vertices() -> None:
    positions = _sample_positions(
        points=[[2.0, 2.0], [2.0, 2.0], [20.0, 2.0]],
        sample_spacing=100,
    )

    assert positions == pytest.approx([0.0, 1.0])


def test_sample_positions_do_not_treat_smooth_curve_tessellation_as_vertices() -> None:
    points = [[float(x), 20.0] for x in np.linspace(0.0, 1000.0, 129)]

    positions = _sample_positions(points=points, sample_spacing=20.0)

    assert len(positions) == 51
    assert positions == pytest.approx(np.linspace(0.0, 1.0, 51))


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


def test_visibility_distinguishes_glare_background_from_a_dark_stripe() -> None:
    image = np.full((80, 112), 220, dtype=np.uint8)
    image[10:17, 8:104] = 20
    image[40:47, 8:104] = 255

    dark = measure_line_profile(
        image,
        [[8.0, 13.0], [103.0, 13.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    ).samples[2]
    bright = measure_line_profile(
        image,
        [[8.0, 43.0], [103.0, 43.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    ).samples[2]

    assert dark.visibility > bright.visibility
    assert bright.visibility < 0.75


def test_visibility_changes_with_bright_line_intensity_along_the_path() -> None:
    image = np.zeros((48, 128), dtype=np.uint8)
    intensity = np.linspace(40, 240, 112).astype(np.uint8)
    image[20:25, 8:120] = intensity

    result = measure_line_profile(
        image,
        [[8.0, 22.0], [119.0, 22.0]],
        parameters=MeasurementParameters(sample_spacing=20, search_radius=12),
    )
    visibility = [sample.visibility for sample in result.samples]

    assert visibility[0] < visibility[-1]
    assert max(visibility) - min(visibility) > 0.5


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


def test_isolated_reliable_width_outlier_uses_neighbor_recommendation() -> None:
    samples = (
        MeasurementSample(0.0, 4.0, 1.0, 0.9),
        MeasurementSample(0.5, 14.0, 1.0, 0.9),
        MeasurementSample(1.0, 4.0, 1.0, 0.9),
    )

    recommended = _recommend_neighbor_widths(samples=samples)

    assert recommended[1].width == pytest.approx(4.0)
    assert recommended[1].confidence == 0.49
    assert recommended[1].reason == "width_outlier"


def test_isolated_narrow_width_uses_consistent_neighbouring_widths() -> None:
    samples = (
        MeasurementSample(0.0, 5.5, 1.0, 0.9),
        MeasurementSample(0.5, 2.0, 1.0, 0.9),
        MeasurementSample(1.0, 4.5, 1.0, 0.9),
    )

    recommended = _recommend_neighbor_widths(samples=samples)

    assert recommended[1].width == pytest.approx(5.0)
    assert recommended[1].reason == "width_outlier"


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
