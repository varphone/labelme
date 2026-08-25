from __future__ import annotations

import numpy as np
import pytest

from labelme._line_profile import LineProfile
from labelme._line_profile import WidthAnchor
from labelme._line_profile_sequence import compare_frames
from labelme._line_profile_sequence import frame_sort_key
from labelme._line_profile_sequence import transfer_frame_profiles
from labelme._line_profile_sequence import validate_frame_mapping
from labelme._shape import Shape


def _shape(label: str, x: float, profile: LineProfile | None) -> Shape:
    return Shape(
        label=label,
        shape_type="linestrip",
        points=np.array([(x, 0.0), (x + 10.0, 0.0)]),
        line_profile=profile,
    )


def test_frame_mapping_validates_and_transfers_only_profile_data() -> None:
    profile = LineProfile(
        width_anchors=(WidthAnchor(0.0, 4.0, "manual", 1.0, True),)
    )
    previous = [_shape("stripe", 0.0, profile)]
    current = [_shape("stripe", 1.0, None)]

    transferred = transfer_frame_profiles(previous, current)

    assert transferred[0].points[0, 0] == 1.0
    assert transferred[0].line_profile == profile
    assert current[0].line_profile is None


def test_frame_mapping_rejects_count_label_path_and_profile_mode_mismatch() -> None:
    previous = [_shape("stripe", 0.0, LineProfile())]
    with pytest.raises(ValueError, match="counts"):
        validate_frame_mapping(previous, [])
    with pytest.raises(ValueError, match="label"):
        validate_frame_mapping(previous, [_shape("other", 0.0, LineProfile())])


def test_frame_comparison_reports_centerline_and_curve_differences() -> None:
    previous = [
        _shape(
            "stripe",
            0.0,
            LineProfile(
                width_anchors=(WidthAnchor(0.0, 4.0, "auto", 0.8, False),)
            ),
        )
    ]
    current = [
        _shape(
            "stripe",
            3.0,
            LineProfile(
                width_anchors=(WidthAnchor(0.0, 6.0, "auto", 0.8, False),)
            ),
        )
    ]

    difference = compare_frames(previous, current)[0]

    assert difference.centerline_max_displacement == pytest.approx(3.0)
    assert difference.width_max_difference == pytest.approx(2.0)
    assert difference.visibility_max_difference is None


def test_frame_sort_key_is_stable_and_case_insensitive() -> None:
    paths = ["frame-10.png", "frame-2.png", "Frame-1.png"]

    assert sorted(paths, key=frame_sort_key) == [
        "Frame-1.png",
        "frame-10.png",
        "frame-2.png",
    ]
