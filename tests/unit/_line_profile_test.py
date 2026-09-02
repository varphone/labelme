from __future__ import annotations

from collections.abc import Callable

import pytest

from labelme._line_profile import LineProfile
from labelme._line_profile import WidthAnchor
from labelme._line_profile import crop_profile
from labelme._line_profile import cumulative_lengths
from labelme._line_profile import extend_profile
from labelme._line_profile import insert_visibility_anchor
from labelme._line_profile import insert_width_anchor
from labelme._line_profile import merge_profiles
from labelme._line_profile import migrate_line_profile_json
from labelme._line_profile import point_to_position
from labelme._line_profile import position_to_point
from labelme._line_profile import profile_boundary_points
from labelme._line_profile import remap_profile
from labelme._line_profile import remove_visibility_anchor
from labelme._line_profile import remove_width_anchor
from labelme._line_profile import resample_profile
from labelme._line_profile import reverse_profile
from labelme._line_profile import split_profile


def _profile_json() -> dict:
    return {
        "schema_version": 1,
        "path_mode": "continuous",
        "parameterization": "normalized_arc_length",
        "width_anchors": [
            {
                "position": 0.0,
                "width": 4.0,
                "source": "auto",
                "confidence": 0.8,
                "confirmed": False,
            },
            {
                "position": 1.0,
                "width": 8.0,
                "source": "manual",
                "confidence": 1.0,
                "confirmed": True,
            },
        ],
        "visibility_anchors": [
            {
                "position": 0.0,
                "visibility": 1.0,
                "source": "auto",
                "confidence": 0.7,
                "confirmed": False,
            },
            {
                "position": 1.0,
                "visibility": 0.5,
                "source": "manual",
                "confidence": 1.0,
                "confirmed": True,
            },
        ],
        "min_width": 2.0,
        "max_width": 10.0,
        "measurement_version": "test-v1",
        "reviewed": True,
    }


def test_line_profile_round_trips_json() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    assert profile.to_json_obj() == _profile_json()


def test_line_profile_interpolates_and_clamps() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    assert profile.evaluate_width(0.5) == pytest.approx(6.0)
    assert profile.evaluate_width(0.0) == pytest.approx(4.0)
    assert profile.evaluate_width(1.0) == pytest.approx(8.0)
    assert profile.evaluate_visibility(0.5) == pytest.approx(0.75)


def test_line_profile_without_anchors_returns_no_measurement() -> None:
    raw = _profile_json()
    raw["width_anchors"] = []
    raw["visibility_anchors"] = []

    profile = LineProfile.from_json_obj(raw)

    assert profile.evaluate_width(0.5) is None
    assert profile.evaluate_visibility(0.5) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("path_mode", "invalid"),
        ("parameterization", "pixel_distance"),
        ("reviewed", "yes"),
    ],
)
def test_line_profile_rejects_unsupported_fields(field: str, value: object) -> None:
    raw = _profile_json()
    raw[field] = value

    with pytest.raises(ValueError):
        LineProfile.from_json_obj(raw)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw["width_anchors"][0].update({"position": 1.1}),
        lambda raw: raw["width_anchors"][1].update({"position": 0.0}),
        lambda raw: raw["width_anchors"][0].update({"width": 0.0}),
        lambda raw: raw["width_anchors"][0].update({"confidence": 2.0}),
        lambda raw: raw["visibility_anchors"][0].update({"visibility": -0.1}),
        lambda raw: raw.update({"min_width": 9.0, "max_width": 8.0}),
    ],
)
def test_line_profile_rejects_invalid_anchor_data(
    mutator: Callable[[dict], None],
) -> None:
    raw = _profile_json()
    mutator(raw)

    with pytest.raises(ValueError):
        LineProfile.from_json_obj(raw)


def test_line_profile_requires_an_object() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        LineProfile.from_json_obj([])


def test_polyline_arc_mapping_handles_zero_length_segments() -> None:
    points = [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]

    assert cumulative_lengths(points) == (0.0, 0.0, 10.0, 20.0)
    assert position_to_point(points, 0.5) == pytest.approx((10.0, 0.0))
    assert point_to_position(points, [10.0, 5.0]) == pytest.approx(0.75)


def test_remap_profile_projects_anchors_after_vertex_edit() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    remapped = remap_profile(
        profile,
        old_points=[[0.0, 0.0], [10.0, 0.0]],
        new_points=[[0.0, 0.0], [20.0, 0.0]],
    )

    assert [anchor.position for anchor in remapped.width_anchors] == [0.0, 0.5]
    assert [anchor.width for anchor in remapped.width_anchors] == [4.0, 8.0]


def test_remap_profile_keeps_translation_positions_unchanged() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    remapped = remap_profile(
        profile,
        old_points=[[0.0, 0.0], [10.0, 0.0]],
        new_points=[[10.0, 20.0], [20.0, 20.0]],
    )

    assert remapped == profile


def test_reverse_split_and_merge_profiles() -> None:
    profile = LineProfile.from_json_obj(_profile_json())
    reversed_profile = reverse_profile(profile)
    left, right = split_profile(profile, split_position=0.5)
    merged = merge_profiles(
        left,
        right,
        left_length=10.0,
        right_length=10.0,
    )

    assert [anchor.position for anchor in reversed_profile.width_anchors] == [0.0, 1.0]
    assert [anchor.position for anchor in left.width_anchors] == [0.0, 1.0]
    assert [anchor.position for anchor in right.width_anchors] == [0.0, 1.0]
    assert merged is not None
    assert [anchor.position for anchor in merged.width_anchors] == [0.0, 0.5, 1.0]


def test_profile_boundary_points_use_full_width_and_local_normals() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    left, right = profile_boundary_points(
        profile,
        [[0.0, 0.0], [10.0, 0.0]],
        samples=3,
    )

    assert left[1] == pytest.approx((5.0, 3.0))
    assert right[1] == pytest.approx((5.0, -3.0))


def test_profile_boundary_points_miter_a_right_angle_corner() -> None:
    profile = LineProfile(
        width_anchors=(
            # Keep the width constant so the corner geometry is isolated.
            WidthAnchor(0.0, 6.0, "manual", 1.0, True),
            WidthAnchor(1.0, 6.0, "manual", 1.0, True),
        )
    )

    left, right = profile_boundary_points(
        profile,
        [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
        samples=9,
    )

    assert left[4] == pytest.approx((7.0, 3.0))
    assert right[4] == pytest.approx((13.0, -3.0))


def test_profile_crop_extend_and_resample_preserve_curve_values() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    cropped = crop_profile(profile, 0.25, 0.75)
    extended = extend_profile(cropped, 0.25, 0.75)
    resampled = resample_profile(profile, [0.0, 0.25, 0.5, 0.75, 1.0])

    assert cropped.evaluate_width(0.0) == pytest.approx(5.0)
    assert cropped.evaluate_width(1.0) == pytest.approx(7.0)
    assert extended.evaluate_width(0.0) == pytest.approx(5.0)
    assert extended.evaluate_width(1.0) == pytest.approx(7.0)
    assert resampled.evaluate_width(0.25) == pytest.approx(5.0)
    assert resampled.evaluate_width(0.75) == pytest.approx(7.0)


def test_profile_schema_migration_is_explicit_and_idempotent() -> None:
    raw = _profile_json()

    migrated = migrate_line_profile_json(raw)

    assert migrated == raw
    assert migrated is not raw
    with pytest.raises(ValueError, match="no line_profile migration"):
        migrate_line_profile_json({**raw, "schema_version": 2})


def test_profile_anchor_insert_and_remove_preserve_interpolated_values() -> None:
    profile = LineProfile.from_json_obj(_profile_json())

    inserted = insert_width_anchor(profile, 0.5)
    inserted = insert_visibility_anchor(inserted, 0.5)

    assert inserted.evaluate_width(0.5) == pytest.approx(6.0)
    assert inserted.evaluate_visibility(0.5) == pytest.approx(0.75)
    assert len(inserted.width_anchors) == 3
    assert len(inserted.visibility_anchors) == 3
    assert len(remove_width_anchor(inserted, 1).width_anchors) == 2
    assert len(remove_visibility_anchor(inserted, 1).visibility_anchors) == 2

    with pytest.raises(ValueError, match="already exists"):
        insert_width_anchor(inserted, 0.5)


def test_profile_measurement_overrides_round_trip_and_validate() -> None:
    raw = _profile_json()
    raw["measurement_overrides"] = {
        "sample_spacing": 4.0,
        "contrast_factor": 0.6,
    }

    profile = LineProfile.from_json_obj(raw)

    assert dict(profile.measurement_overrides) == raw["measurement_overrides"]
    assert profile.to_json_obj()["measurement_overrides"] == raw[
        "measurement_overrides"
    ]
    with pytest.raises(ValueError, match="unsupported measurement override"):
        LineProfile.from_json_obj(
            {**_profile_json(), "measurement_overrides": {"x": 1}}
        )
    with pytest.raises(ValueError, match="must be positive"):
        LineProfile.from_json_obj(
            {**_profile_json(), "measurement_overrides": {"min_width": 0}}
        )
    with pytest.raises(ValueError, match="duplicate"):
        LineProfile(
            measurement_overrides=(
                ("sample_spacing", 4.0),
                ("sample_spacing", 5.0),
            )
        )
