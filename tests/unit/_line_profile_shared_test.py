from __future__ import annotations

import pytest

from labelme._line_profile import LineProfile
from labelme._line_profile import ProfileAnchor
from labelme._line_profile import ProfileValue
from labelme._line_profile import evaluate_profile
from labelme._line_profile import remove_profile_property
from labelme._line_profile import reverse_profile
from labelme._line_profile import split_profile
from labelme._line_profile import update_profile_anchor


def _profile() -> LineProfile:
    return LineProfile(
        anchors=(
            ProfileAnchor(
                0.0,
                width=ProfileValue(4.0, "manual", 1.0, True),
                visibility=ProfileValue(1.0, "manual", 1.0, True),
            ),
            ProfileAnchor(
                0.5,
                visibility=ProfileValue(0.25, "auto", 0.4, False),
            ),
        )
    )


def test_shared_anchor_json_round_trip_keeps_independent_properties() -> None:
    profile = _profile()
    raw = profile.to_json_obj()

    assert raw["schema_version"] == 2
    assert len(raw["anchors"]) == 2
    assert "width_anchors" not in raw
    assert "visibility_anchors" not in raw
    assert raw["anchors"][0]["width"]["value"] == 4.0
    assert raw["anchors"][1]["width"] is None
    assert LineProfile.from_json_obj(raw) == profile


def test_shared_anchor_properties_interpolate_independently() -> None:
    width, visibility = evaluate_profile(_profile(), 0.25)

    assert width == pytest.approx(4.0)
    assert visibility == pytest.approx(0.625)


def test_removing_one_property_keeps_shared_anchor() -> None:
    profile = remove_profile_property(_profile(), 0, "width")

    assert len(profile.anchors) == 2
    assert profile.anchors[0].width is None
    assert profile.anchors[0].visibility is not None


def test_updating_shared_anchor_preserves_other_property() -> None:
    profile = update_profile_anchor(
        _profile(),
        0,
        position=0.25,
        width=ProfileValue(8.0, "manual", 1.0, True),
    )

    assert profile.anchors[0].position == pytest.approx(0.25)
    assert profile.anchors[0].width is not None
    assert profile.anchors[0].width.value == 8.0
    assert profile.anchors[0].visibility is not None


def test_geometry_transforms_move_shared_positions_once() -> None:
    profile = _profile()

    reversed_profile = reverse_profile(profile)
    left, right = split_profile(profile, 0.5)

    assert reversed_profile.anchors[0].position == pytest.approx(0.5)
    assert reversed_profile.anchors[0].visibility is not None
    assert left.anchors[0].width is not None
    assert left.anchors[0].visibility is not None
    assert right.anchors[0].visibility is not None
