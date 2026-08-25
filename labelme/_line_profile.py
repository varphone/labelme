from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import TypedDict
from typing import cast

PathMode = Literal["continuous", "visible_only"]
Parameterization = Literal["normalized_arc_length"]
AnchorSource = Literal["auto", "manual"]

CURRENT_SCHEMA_VERSION = 1
DEFAULT_PARAMETERIZATION: Parameterization = "normalized_arc_length"


class WidthAnchorDict(TypedDict):
    position: float
    width: float
    source: AnchorSource
    confidence: float
    confirmed: bool


class VisibilityAnchorDict(TypedDict):
    position: float
    visibility: float
    source: AnchorSource
    confidence: float
    confirmed: bool


class LineProfileDict(TypedDict):
    schema_version: int
    path_mode: PathMode
    parameterization: Parameterization
    width_anchors: list[WidthAnchorDict]
    visibility_anchors: list[VisibilityAnchorDict]
    min_width: float | None
    max_width: float | None
    measurement_version: str | None
    reviewed: bool


@dataclasses.dataclass(frozen=True)
class WidthAnchor:
    position: float
    width: float
    source: AnchorSource
    confidence: float
    confirmed: bool


@dataclasses.dataclass(frozen=True)
class VisibilityAnchor:
    position: float
    visibility: float
    source: AnchorSource
    confidence: float
    confirmed: bool


Anchor = WidthAnchor | VisibilityAnchor


@dataclasses.dataclass(frozen=True)
class LineProfile:
    schema_version: int = CURRENT_SCHEMA_VERSION
    path_mode: PathMode = "continuous"
    parameterization: Parameterization = DEFAULT_PARAMETERIZATION
    width_anchors: tuple[WidthAnchor, ...] = ()
    visibility_anchors: tuple[VisibilityAnchor, ...] = ()
    min_width: float | None = None
    max_width: float | None = None
    measurement_version: str | None = None
    reviewed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise ValueError("schema_version must be an int")
        if self.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported line_profile schema_version: {self.schema_version}"
            )
        if self.path_mode not in ("continuous", "visible_only"):
            raise ValueError(f"path_mode is unsupported: {self.path_mode!r}")
        if self.parameterization != DEFAULT_PARAMETERIZATION:
            raise ValueError(
                "parameterization must be 'normalized_arc_length': "
                f"{self.parameterization!r}"
            )
        if not isinstance(self.reviewed, bool):
            raise ValueError("reviewed must be bool")
        if self.measurement_version is not None and not isinstance(
            self.measurement_version, str
        ):
            raise ValueError("measurement_version must be str or null")

        min_width = _optional_positive_float(self.min_width, "min_width")
        max_width = _optional_positive_float(self.max_width, "max_width")
        if min_width is not None and max_width is not None and min_width > max_width:
            raise ValueError("min_width must not exceed max_width")
        for anchor in self.width_anchors:
            _validate_width_anchor(anchor=anchor)
            if min_width is not None and anchor.width < min_width:
                raise ValueError("width anchor is below min_width")
            if max_width is not None and anchor.width > max_width:
                raise ValueError("width anchor exceeds max_width")
        for anchor in self.visibility_anchors:
            _validate_visibility_anchor(anchor=anchor)
        _validate_positions(
            positions=[anchor.position for anchor in self.width_anchors],
            field="width_anchors",
        )
        _validate_positions(
            positions=[anchor.position for anchor in self.visibility_anchors],
            field="visibility_anchors",
        )

    @classmethod
    def from_json_obj(cls, value: object) -> LineProfile:
        if not isinstance(value, dict):
            raise ValueError("line_profile must be an object")

        schema_version = _required_int(value, "schema_version")
        path_mode = _required_literal(
            value, "path_mode", ("continuous", "visible_only")
        )
        parameterization = _required_literal(
            value, "parameterization", (DEFAULT_PARAMETERIZATION,)
        )
        width_anchors = tuple(
            _width_anchor_from_json(item, index=index)
            for index, item in enumerate(_required_list(value, "width_anchors"))
        )
        visibility_anchors = tuple(
            _visibility_anchor_from_json(item, index=index)
            for index, item in enumerate(
                _required_list(value, "visibility_anchors")
            )
        )
        min_width = _optional_number(value, "min_width")
        max_width = _optional_number(value, "max_width")
        measurement_version = value.get("measurement_version")
        if measurement_version is not None and not isinstance(
            measurement_version, str
        ):
            raise ValueError("measurement_version must be str or null")
        reviewed = value.get("reviewed")
        if not isinstance(reviewed, bool):
            raise ValueError("reviewed must be bool")
        return cls(
            schema_version=schema_version,
            path_mode=cast(PathMode, path_mode),
            parameterization=cast(Parameterization, parameterization),
            width_anchors=width_anchors,
            visibility_anchors=visibility_anchors,
            min_width=min_width,
            max_width=max_width,
            measurement_version=measurement_version,
            reviewed=reviewed,
        )

    def to_json_obj(self) -> LineProfileDict:
        return {
            "schema_version": self.schema_version,
            "path_mode": self.path_mode,
            "parameterization": self.parameterization,
            "width_anchors": [
                {
                    "position": anchor.position,
                    "width": anchor.width,
                    "source": anchor.source,
                    "confidence": anchor.confidence,
                    "confirmed": anchor.confirmed,
                }
                for anchor in self.width_anchors
            ],
            "visibility_anchors": [
                {
                    "position": anchor.position,
                    "visibility": anchor.visibility,
                    "source": anchor.source,
                    "confidence": anchor.confidence,
                    "confirmed": anchor.confirmed,
                }
                for anchor in self.visibility_anchors
            ],
            "min_width": self.min_width,
            "max_width": self.max_width,
            "measurement_version": self.measurement_version,
            "reviewed": self.reviewed,
        }

    def evaluate_width(self, position: float) -> float | None:
        _validate_position(position, field="position")
        return _interpolate(
            position=position,
            anchors=[(anchor.position, anchor.width) for anchor in self.width_anchors],
        )

    def evaluate_visibility(self, position: float) -> float | None:
        _validate_position(position, field="position")
        return _interpolate(
            position=position,
            anchors=[
                (anchor.position, anchor.visibility)
                for anchor in self.visibility_anchors
            ],
        )


def cumulative_lengths(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Return cumulative pixel lengths at each polyline vertex."""
    if len(points) == 0:
        raise ValueError("points must not be empty")
    result = [0.0]
    for left, right in zip(points, points[1:]):
        if len(left) != 2 or len(right) != 2:
            raise ValueError("points must contain [x, y] pairs")
        result.append(result[-1] + math.hypot(right[0] - left[0], right[1] - left[1]))
    return tuple(result)


def position_to_point(
    points: Sequence[Sequence[float]], position: float
) -> tuple[float, float]:
    """Map normalized arc length to a point, skipping zero-length segments."""
    _validate_position(position, field="position")
    lengths = cumulative_lengths(points)
    total = lengths[-1]
    if total == 0:
        return float(points[0][0]), float(points[0][1])
    distance = position * total
    for index, (start, end) in enumerate(zip(lengths, lengths[1:])):
        if distance <= end or index == len(points) - 2:
            segment_length = end - start
            fraction = (
                0.0
                if segment_length == 0
                else (distance - start) / segment_length
            )
            return (
                float(
                    points[index][0]
                    + fraction * (points[index + 1][0] - points[index][0])
                ),
                float(
                    points[index][1]
                    + fraction * (points[index + 1][1] - points[index][1])
                ),
            )
    return float(points[-1][0]), float(points[-1][1])


def point_to_position(
    points: Sequence[Sequence[float]], point: Sequence[float]
) -> float:
    """Project a point onto a polyline and return its normalized arc position."""
    if len(point) != 2:
        raise ValueError("point must contain [x, y]")
    lengths = cumulative_lengths(points)
    total = lengths[-1]
    if total == 0:
        return 0.0
    best_distance = math.inf
    best_distance_along = 0.0
    px, py = point
    for index, (start, end) in enumerate(zip(lengths, lengths[1:])):
        x0, y0 = points[index]
        dx, dy = points[index + 1][0] - x0, points[index + 1][1] - y0
        segment_length = end - start
        if segment_length == 0:
            fraction = 0.0
        else:
            fraction = max(
                0.0,
                min(1.0, ((px - x0) * dx + (py - y0) * dy) / (segment_length**2)),
            )
        projected_x, projected_y = x0 + fraction * dx, y0 + fraction * dy
        distance = (px - projected_x) ** 2 + (py - projected_y) ** 2
        if distance < best_distance:
            best_distance = distance
            best_distance_along = start + fraction * segment_length
    return best_distance_along / total


def profile_boundary_points(
    profile: LineProfile,
    points: Sequence[Sequence[float]],
    *,
    samples: int = 64,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    """Sample left and right boundaries without creating annotation Shapes."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for index in range(samples):
        position = index / (samples - 1)
        width = profile.evaluate_width(position)
        center = position_to_point(points, position)
        if width is None:
            left.append(center)
            right.append(center)
            continue
        before = position_to_point(points, max(0.0, position - 1e-4))
        after = position_to_point(points, min(1.0, position + 1e-4))
        dx, dy = after[0] - before[0], after[1] - before[1]
        length = math.hypot(dx, dy)
        if length == 0:
            dx, dy = _fallback_tangent(points)
            length = math.hypot(dx, dy)
        if length == 0:
            left.append(center)
            right.append(center)
            continue
        nx, ny = -dy / length, dx / length
        half_width = width / 2.0
        left.append((center[0] + nx * half_width, center[1] + ny * half_width))
        right.append((center[0] - nx * half_width, center[1] - ny * half_width))
    return tuple(left), tuple(right)


def remap_profile(
    profile: LineProfile,
    old_points: Sequence[Sequence[float]],
    new_points: Sequence[Sequence[float]],
) -> LineProfile:
    """Remap anchors by projecting their old physical positions onto a new path."""
    if len(old_points) == 0 or len(new_points) == 0:
        raise ValueError("old_points and new_points must not be empty")
    if _is_translation(old_points=old_points, new_points=new_points):
        return profile
    width_anchors = _remap_width_anchors(profile.width_anchors, old_points, new_points)
    visibility_anchors = _remap_visibility_anchors(
        profile.visibility_anchors, old_points, new_points
    )
    return dataclasses.replace(
        profile,
        width_anchors=width_anchors,
        visibility_anchors=visibility_anchors,
    )


def reverse_profile(profile: LineProfile) -> LineProfile:
    return dataclasses.replace(
        profile,
        width_anchors=tuple(
            dataclasses.replace(anchor, position=1.0 - anchor.position)
            for anchor in reversed(profile.width_anchors)
        ),
        visibility_anchors=tuple(
            dataclasses.replace(anchor, position=1.0 - anchor.position)
            for anchor in reversed(profile.visibility_anchors)
        ),
    )


def split_profile(
    profile: LineProfile, split_position: float
) -> tuple[LineProfile, LineProfile]:
    """Split a profile at a normalized position and normalize both halves."""
    _validate_position(split_position, field="split_position")
    if split_position <= 0.0 or split_position >= 1.0:
        raise ValueError("split_position must be strictly within (0, 1)")
    return (
        dataclasses.replace(
            profile,
            width_anchors=_split_width_anchors(profile, split_position, left=True),
            visibility_anchors=_split_visibility_anchors(
                profile, split_position, left=True
            ),
        ),
        dataclasses.replace(
            profile,
            width_anchors=_split_width_anchors(profile, split_position, left=False),
            visibility_anchors=_split_visibility_anchors(
                profile, split_position, left=False
            ),
        ),
    )


def merge_profiles(
    left: LineProfile | None,
    right: LineProfile | None,
    *,
    left_length: float,
    right_length: float,
    join_policy: Literal["left", "right", "average"] = "left",
) -> LineProfile | None:
    """Concatenate profiles using physical path lengths for position mapping."""
    if left is None:
        return (
            None
            if right is None
            else _scale_profile(
                right,
                offset=left_length / (left_length + right_length),
                scale=right_length / (left_length + right_length),
            )
        )
    if right is None:
        return _scale_profile(
            left,
            offset=0.0,
            scale=left_length / (left_length + right_length),
        )
    if left.path_mode != right.path_mode:
        raise ValueError("cannot merge profiles with different path_mode")
    if left_length <= 0 or right_length <= 0:
        raise ValueError("profile path lengths must be positive")
    if join_policy not in ("left", "right", "average"):
        raise ValueError(f"unsupported join_policy: {join_policy!r}")
    total = left_length + right_length
    left_scale = left_length / total
    right_offset = left_scale
    right_scale = right_length / total
    width = _merge_anchor_sets(
        left.width_anchors,
        right.width_anchors,
        left_scale=left_scale,
        right_offset=right_offset,
        right_scale=right_scale,
        join_policy=join_policy,
    )
    visibility = _merge_anchor_sets(
        left.visibility_anchors,
        right.visibility_anchors,
        left_scale=left_scale,
        right_offset=right_offset,
        right_scale=right_scale,
        join_policy=join_policy,
    )
    return dataclasses.replace(
        left,
        width_anchors=width,
        visibility_anchors=visibility,
        min_width=_merge_optional_min(left.min_width, right.min_width),
        max_width=_merge_optional_max(left.max_width, right.max_width),
        measurement_version=(
            left.measurement_version
            if left.measurement_version == right.measurement_version
            else None
        ),
        reviewed=left.reviewed and right.reviewed,
    )


def _is_translation(
    *, old_points: Sequence[Sequence[float]], new_points: Sequence[Sequence[float]]
) -> bool:
    if len(old_points) != len(new_points):
        return False
    dx = new_points[0][0] - old_points[0][0]
    dy = new_points[0][1] - old_points[0][1]
    return all(
        math.isclose(new[0] - old[0], dx, abs_tol=1e-9)
        and math.isclose(new[1] - old[1], dy, abs_tol=1e-9)
        for old, new in zip(old_points, new_points)
    )


def _fallback_tangent(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    for left, right in zip(points, points[1:]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        if dx != 0 or dy != 0:
            return float(dx), float(dy)
    return 0.0, 0.0


def _scale_profile(profile: LineProfile, *, offset: float, scale: float) -> LineProfile:
    return dataclasses.replace(
        profile,
        width_anchors=tuple(
            dataclasses.replace(anchor, position=offset + anchor.position * scale)
            for anchor in profile.width_anchors
        ),
        visibility_anchors=tuple(
            dataclasses.replace(anchor, position=offset + anchor.position * scale)
            for anchor in profile.visibility_anchors
        ),
    )


def _remap_width_anchors(
    anchors: tuple[WidthAnchor, ...],
    old_points: Sequence[Sequence[float]],
    new_points: Sequence[Sequence[float]],
) -> tuple[WidthAnchor, ...]:
    remapped = [
        dataclasses.replace(
            anchor,
            position=point_to_position(
                new_points, position_to_point(old_points, anchor.position)
            ),
        )
        for anchor in anchors
    ]
    return _deduplicate_anchors(remapped)


def _remap_visibility_anchors(
    anchors: tuple[VisibilityAnchor, ...],
    old_points: Sequence[Sequence[float]],
    new_points: Sequence[Sequence[float]],
) -> tuple[VisibilityAnchor, ...]:
    remapped = [
        dataclasses.replace(
            anchor,
            position=point_to_position(
                new_points, position_to_point(old_points, anchor.position)
            ),
        )
        for anchor in anchors
    ]
    return _deduplicate_anchors(remapped)


def _deduplicate_anchors(anchors: list[Anchor]) -> tuple[Anchor, ...]:
    result: list[Anchor] = []
    for anchor in sorted(anchors, key=lambda item: item.position):
        if result and math.isclose(result[-1].position, anchor.position, abs_tol=1e-9):
            if _anchor_rank(anchor) > _anchor_rank(result[-1]):
                result[-1] = anchor
        else:
            result.append(anchor)
    return tuple(result)


def _anchor_rank(anchor: Anchor) -> tuple[int, int, float]:
    return (
        int(anchor.confirmed),
        int(anchor.source == "manual"),
        anchor.confidence,
    )


def _split_width_anchors(
    profile: LineProfile, split_position: float, *, left: bool
) -> tuple[WidthAnchor, ...]:
    return _split_anchors(
        anchors=profile.width_anchors,
        evaluate=profile.evaluate_width,
        split_position=split_position,
        left=left,
        make_anchor=WidthAnchor,
    )


def _split_visibility_anchors(
    profile: LineProfile, split_position: float, *, left: bool
) -> tuple[VisibilityAnchor, ...]:
    return _split_anchors(
        anchors=profile.visibility_anchors,
        evaluate=profile.evaluate_visibility,
        split_position=split_position,
        left=left,
        make_anchor=VisibilityAnchor,
    )


def _split_anchors(
    *,
    anchors: tuple[Anchor, ...],
    evaluate: Callable[[float], float | None],
    split_position: float,
    left: bool,
    make_anchor: type[WidthAnchor] | type[VisibilityAnchor],
) -> tuple[Anchor, ...]:
    if not anchors:
        return ()
    start, end = (0.0, split_position) if left else (split_position, 1.0)
    selected = [anchor for anchor in anchors if start <= anchor.position <= end]
    for boundary in (start, end):
        if not any(math.isclose(anchor.position, boundary) for anchor in selected):
            value = evaluate(boundary)
            if value is not None:
                kwargs = {
                    "position": boundary,
                    "source": "auto",
                    "confidence": 0.0,
                    "confirmed": False,
                }
                if make_anchor is WidthAnchor:
                    selected.append(WidthAnchor(width=value, **kwargs))
                else:
                    selected.append(VisibilityAnchor(visibility=value, **kwargs))
    scale = split_position if left else 1.0 - split_position
    return _deduplicate_anchors(
        [
            dataclasses.replace(
                anchor,
                position=(anchor.position if left else anchor.position - split_position)
                / scale,
            )
            for anchor in selected
        ]
    )


def _merge_anchor_sets(
    left: tuple[Anchor, ...],
    right: tuple[Anchor, ...],
    *,
    left_scale: float,
    right_offset: float,
    right_scale: float,
    join_policy: Literal["left", "right", "average"],
) -> tuple[Anchor, ...]:
    mapped_left = [
        dataclasses.replace(anchor, position=anchor.position * left_scale)
        for anchor in left
    ]
    mapped_right = [
        dataclasses.replace(
            anchor, position=right_offset + anchor.position * right_scale
        )
        for anchor in right
    ]
    if not mapped_left:
        return tuple(mapped_right)
    if not mapped_right:
        return tuple(mapped_left)
    at_join_left = [
        anchor for anchor in mapped_left if math.isclose(anchor.position, left_scale)
    ]
    at_join_right = [
        anchor
        for anchor in mapped_right
        if math.isclose(anchor.position, right_offset)
    ]
    if at_join_left and at_join_right:
        left_anchor, right_anchor = at_join_left[-1], at_join_right[0]
        if join_policy == "right":
            mapped_left.remove(left_anchor)
        elif join_policy == "average":
            value_name = "width" if hasattr(left_anchor, "width") else "visibility"
            average = (
                getattr(left_anchor, value_name)
                + getattr(right_anchor, value_name)
            ) / 2
            mapped_left[-1] = dataclasses.replace(
                left_anchor,
                **{
                    value_name: average,
                    "confidence": min(
                        left_anchor.confidence, right_anchor.confidence
                    ),
                },
            )
            mapped_right.remove(right_anchor)
        else:
            mapped_right.remove(right_anchor)
    return _deduplicate_anchors(mapped_left + mapped_right)


def _merge_optional_min(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _merge_optional_max(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _required_list(value: dict[str, Any], field: str) -> list[object]:
    result = value.get(field)
    if not isinstance(result, list):
        raise ValueError(f"{field} must be a list")
    return result


def _required_int(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{field} must be int")
    return result


def _required_number(value: dict[str, Any], field: str) -> float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise ValueError(f"{field} must be a number")
    result = float(result)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_number(value: dict[str, Any], field: str) -> float | None:
    result = value.get(field)
    if result is None:
        return None
    return _required_number(value, field)


def _optional_positive_float(value: float | None, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a positive number or null")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be a positive finite number or null")
    return value


def _required_literal(
    value: dict[str, Any], field: str, choices: tuple[str, ...]
) -> str:
    result = value.get(field)
    if not isinstance(result, str) or result not in choices:
        raise ValueError(f"{field} must be one of {choices!r}")
    return result


def _required_anchor_dict(value: object, *, field: str, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field}[{index}] must be an object")
    return value


def _width_anchor_from_json(value: object, *, index: int) -> WidthAnchor:
    value = _required_anchor_dict(value, field="width_anchors", index=index)
    source = _required_literal(value, "source", ("auto", "manual"))
    confirmed = value.get("confirmed")
    if not isinstance(confirmed, bool):
        raise ValueError(f"width_anchors[{index}].confirmed must be bool")
    return WidthAnchor(
        position=_required_number(value, "position"),
        width=_required_number(value, "width"),
        source=cast(AnchorSource, source),
        confidence=_required_number(value, "confidence"),
        confirmed=confirmed,
    )


def _visibility_anchor_from_json(value: object, *, index: int) -> VisibilityAnchor:
    value = _required_anchor_dict(value, field="visibility_anchors", index=index)
    source = _required_literal(value, "source", ("auto", "manual"))
    confirmed = value.get("confirmed")
    if not isinstance(confirmed, bool):
        raise ValueError(f"visibility_anchors[{index}].confirmed must be bool")
    return VisibilityAnchor(
        position=_required_number(value, "position"),
        visibility=_required_number(value, "visibility"),
        source=cast(AnchorSource, source),
        confidence=_required_number(value, "confidence"),
        confirmed=confirmed,
    )


def _validate_position(position: float, *, field: str) -> None:
    if not math.isfinite(position) or not 0.0 <= position <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")


def _validate_positions(*, positions: list[float], field: str) -> None:
    for index, position in enumerate(positions):
        _validate_position(position, field=f"{field}[{index}].position")
    if any(left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError(f"{field} positions must be strictly increasing")


def _validate_width_anchor(*, anchor: WidthAnchor) -> None:
    _validate_position(anchor.position, field="width anchor position")
    if not math.isfinite(anchor.width) or anchor.width <= 0:
        raise ValueError("width anchor width must be positive and finite")
    _validate_confidence(anchor.confidence)
    _validate_source(anchor.source)
    if not isinstance(anchor.confirmed, bool):
        raise ValueError("width anchor confirmed must be bool")


def _validate_visibility_anchor(*, anchor: VisibilityAnchor) -> None:
    _validate_position(anchor.position, field="visibility anchor position")
    if not math.isfinite(anchor.visibility) or not 0.0 <= anchor.visibility <= 1.0:
        raise ValueError("visibility anchor must be finite and within [0, 1]")
    _validate_confidence(anchor.confidence)
    _validate_source(anchor.source)
    if not isinstance(anchor.confirmed, bool):
        raise ValueError("visibility anchor confirmed must be bool")


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")


def _validate_source(source: str) -> None:
    if source not in ("auto", "manual"):
        raise ValueError(f"source is unsupported: {source!r}")


def _interpolate(
    *, position: float, anchors: list[tuple[float, float]]
) -> float | None:
    if not anchors:
        return None
    if len(anchors) == 1 or position <= anchors[0][0]:
        return anchors[0][1]
    if position >= anchors[-1][0]:
        return anchors[-1][1]
    for (left_position, left_value), (right_position, right_value) in zip(
        anchors, anchors[1:]
    ):
        if position <= right_position:
            fraction = (position - left_position) / (right_position - left_position)
            return left_value + fraction * (right_value - left_value)
    raise AssertionError("position must be covered by the last anchor")
