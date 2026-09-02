from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal
from typing import NotRequired
from typing import TypedDict
from typing import cast

PathMode = Literal["continuous", "visible_only"]
Parameterization = Literal["normalized_arc_length"]
AnchorSource = Literal["auto", "manual"]
MeasurementOverrideKey = Literal[
    "sample_spacing",
    "search_radius",
    "min_width",
    "max_width",
    "contrast_factor",
]
_MEASUREMENT_OVERRIDE_KEYS = frozenset(
    {
        "sample_spacing",
        "search_radius",
        "min_width",
        "max_width",
        "contrast_factor",
    }
)

CURRENT_SCHEMA_VERSION = 2
DEFAULT_PARAMETERIZATION: Parameterization = "normalized_arc_length"


class ProfileValueDict(TypedDict):
    value: float
    source: AnchorSource
    confidence: float
    confirmed: bool


class ProfileAnchorDict(TypedDict):
    position: float
    width: ProfileValueDict | None
    visibility: ProfileValueDict | None


class LineProfileDict(TypedDict):
    schema_version: int
    path_mode: PathMode
    parameterization: Parameterization
    anchors: list[ProfileAnchorDict]
    min_width: float | None
    max_width: float | None
    measurement_version: str | None
    reviewed: bool
    measurement_overrides: NotRequired[dict[str, float]]


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


@dataclasses.dataclass(frozen=True)
class ProfileValue:
    """Metadata and value for one independently measured profile property."""

    value: float
    source: AnchorSource
    confidence: float
    confirmed: bool


@dataclasses.dataclass(frozen=True)
class ProfileAnchor:
    """A shared position carrying optional width and visibility observations."""

    position: float
    width: ProfileValue | None = None
    visibility: ProfileValue | None = None


Anchor = WidthAnchor | VisibilityAnchor


@dataclasses.dataclass(frozen=True, init=False)
class LineProfile:
    schema_version: int = CURRENT_SCHEMA_VERSION
    path_mode: PathMode = "continuous"
    parameterization: Parameterization = DEFAULT_PARAMETERIZATION
    min_width: float | None = None
    max_width: float | None = None
    measurement_version: str | None = None
    reviewed: bool = False
    measurement_overrides: tuple[tuple[MeasurementOverrideKey, float], ...] = ()
    anchors: tuple[ProfileAnchor, ...] = ()

    def __init__(
        self,
        schema_version: int = CURRENT_SCHEMA_VERSION,
        path_mode: PathMode = "continuous",
        parameterization: Parameterization = DEFAULT_PARAMETERIZATION,
        *,
        anchors: Sequence[ProfileAnchor] = (),
        min_width: float | None = None,
        max_width: float | None = None,
        measurement_version: str | None = None,
        reviewed: bool = False,
        measurement_overrides: Sequence[tuple[MeasurementOverrideKey, float]] = (),
        width_anchors: Sequence[WidthAnchor] = (),
        visibility_anchors: Sequence[VisibilityAnchor] = (),
    ) -> None:
        if anchors and (width_anchors or visibility_anchors):
            raise ValueError("use anchors instead of legacy anchor arrays")
        canonical = tuple(anchors) or _merge_legacy_anchors(
            width_anchors, visibility_anchors
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "path_mode", path_mode)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "min_width", min_width)
        object.__setattr__(self, "max_width", max_width)
        object.__setattr__(self, "measurement_version", measurement_version)
        object.__setattr__(self, "reviewed", reviewed)
        object.__setattr__(self, "measurement_overrides", tuple(measurement_overrides))
        object.__setattr__(self, "anchors", canonical)
        self.__post_init__()

    @property
    def width_anchors(self) -> tuple[WidthAnchor, ...]:
        """Compatibility view; the canonical state is ``anchors``."""
        return _legacy_width_anchors(self.anchors)

    @property
    def visibility_anchors(self) -> tuple[VisibilityAnchor, ...]:
        """Compatibility view; the canonical state is ``anchors``."""
        return _legacy_visibility_anchors(self.anchors)

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
        _validate_measurement_overrides(self.measurement_overrides)

        min_width = _optional_positive_float(self.min_width, "min_width")
        max_width = _optional_positive_float(self.max_width, "max_width")
        if min_width is not None and max_width is not None and min_width > max_width:
            raise ValueError("min_width must not exceed max_width")
        _validate_shared_anchors(self.anchors, min_width=min_width, max_width=max_width)

    @classmethod
    def from_json_obj(cls, value: object) -> LineProfile:
        value = migrate_line_profile_json(value)

        schema_version = _required_int(value, "schema_version")
        path_mode = _required_literal(
            value, "path_mode", ("continuous", "visible_only")
        )
        parameterization = _required_literal(
            value, "parameterization", (DEFAULT_PARAMETERIZATION,)
        )
        if schema_version == 2:
            anchors = tuple(
                _profile_anchor_from_json(item, index=index)
                for index, item in enumerate(_required_list(value, "anchors"))
            )
            return cls(
                schema_version=schema_version,
                path_mode=cast(PathMode, path_mode),
                parameterization=cast(Parameterization, parameterization),
                min_width=_optional_number(value, "min_width"),
                max_width=_optional_number(value, "max_width"),
                measurement_version=value.get("measurement_version"),
                reviewed=value.get("reviewed"),
                measurement_overrides=_measurement_overrides_from_json(
                    value.get("measurement_overrides")
                ),
                anchors=anchors,
            )

    def to_json_obj(self) -> LineProfileDict:
        if self.schema_version == 2:
            result: LineProfileDict = {
                "schema_version": self.schema_version,
                "path_mode": self.path_mode,
                "parameterization": self.parameterization,
                "anchors": [_profile_anchor_to_json(anchor) for anchor in self.anchors],
                "min_width": self.min_width,
                "max_width": self.max_width,
                "measurement_version": self.measurement_version,
                "reviewed": self.reviewed,
            }
            if self.measurement_overrides:
                result["measurement_overrides"] = dict(self.measurement_overrides)
            return result
        raise ValueError("unsupported line_profile schema version")

    def evaluate_width(self, position: float) -> float | None:
        _validate_position(position, field="position")
        return _interpolate(
            position=position,
            anchors=[
                (anchor.position, anchor.width.value)
                for anchor in self.anchors
                if anchor.width is not None
            ],
        )

    def evaluate_visibility(self, position: float) -> float | None:
        _validate_position(position, field="position")
        return _interpolate(
            position=position,
            anchors=[
                (anchor.position, anchor.visibility.value)
                for anchor in self.anchors
                if anchor.visibility is not None
            ],
        )

    @property
    def width_anchor_count(self) -> int:
        return sum(anchor.width is not None for anchor in self.anchors)

    @property
    def visibility_anchor_count(self) -> int:
        return sum(anchor.visibility is not None for anchor in self.anchors)


def validate_profile(
    profile: LineProfile,
    shape: object | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """Validate a profile explicitly at application boundaries.

    ``LineProfile`` validates itself on construction; this named function keeps
    validation available to codecs and result-acceptance services without
    duplicating the invariants.
    """
    if not isinstance(profile, LineProfile):
        raise TypeError("profile must be a LineProfile")
    profile.__post_init__()
    if shape is not None:
        shape_type = getattr(shape, "shape_type", None)
        if isinstance(shape, dict):
            shape_type = shape.get("shape_type")
        if shape_type != "linestrip":
            raise ValueError("line_profile is only supported for linestrip")
        points = getattr(shape, "points", None)
        if isinstance(shape, dict):
            points = shape.get("points")
        if points is not None and len(points) < 2:
            raise ValueError("a profile requires a centerline with at least two points")
    if image_size is not None:
        if len(image_size) != 2 or any(size <= 0 for size in image_size):
            raise ValueError("image_size must contain two positive dimensions")


def evaluate_profile(
    profile: LineProfile, position: float
) -> tuple[float | None, float | None]:
    """Evaluate width and visibility independently at one shared position."""
    return profile.evaluate_width(position), profile.evaluate_visibility(position)


def migrate_line_profile_json(value: object) -> dict[str, Any]:
    """Apply explicit, idempotent schema migrations to a JSON object.

    Version 1 is the first published profile schema, so there is intentionally
    no heuristic migration from an earlier shape. Unknown future versions are
    rejected by the model and retained by the annotation codec as raw data.
    """
    if not isinstance(value, dict):
        raise ValueError("line_profile must be an object")
    version = _required_int(value, "schema_version")
    if version != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"no line_profile migration is available for schema_version {version}"
        )
    return dict(value)


def _profile_value_to_json(value: ProfileValue) -> ProfileValueDict:
    return {
        "value": value.value,
        "source": value.source,
        "confidence": value.confidence,
        "confirmed": value.confirmed,
    }


def _profile_anchor_to_json(anchor: ProfileAnchor) -> ProfileAnchorDict:
    return {
        "position": anchor.position,
        "width": None if anchor.width is None else _profile_value_to_json(anchor.width),
        "visibility": (
            None
            if anchor.visibility is None
            else _profile_value_to_json(anchor.visibility)
        ),
    }


def _profile_value_from_json(value: object, *, field: str) -> ProfileValue | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object or null")
    source = _required_literal(value, "source", ("auto", "manual"))
    confirmed = value.get("confirmed")
    if not isinstance(confirmed, bool):
        raise ValueError(f"{field}.confirmed must be bool")
    return ProfileValue(
        value=_required_number(value, "value"),
        source=cast(AnchorSource, source),
        confidence=_required_number(value, "confidence"),
        confirmed=confirmed,
    )


def _profile_anchor_from_json(value: object, *, index: int) -> ProfileAnchor:
    field = f"anchors[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    anchor = ProfileAnchor(
        position=_required_number(value, "position"),
        width=_profile_value_from_json(value.get("width"), field=f"{field}.width"),
        visibility=_profile_value_from_json(
            value.get("visibility"), field=f"{field}.visibility"
        ),
    )
    if anchor.width is None and anchor.visibility is None:
        raise ValueError(f"{field} must contain width or visibility")
    return anchor


def _merge_legacy_anchors(
    width_anchors: Sequence[WidthAnchor],
    visibility_anchors: Sequence[VisibilityAnchor],
) -> tuple[ProfileAnchor, ...]:
    positions = sorted(
        {anchor.position for anchor in (*width_anchors, *visibility_anchors)}
    )
    result: list[ProfileAnchor] = []
    for position in positions:
        width = next(
            (
                a
                for a in width_anchors
                if math.isclose(a.position, position, abs_tol=1e-9)
            ),
            None,
        )
        visibility = next(
            (
                a
                for a in visibility_anchors
                if math.isclose(a.position, position, abs_tol=1e-9)
            ),
            None,
        )
        result.append(
            ProfileAnchor(
                position=position,
                width=None
                if width is None
                else ProfileValue(
                    width.width, width.source, width.confidence, width.confirmed
                ),
                visibility=None
                if visibility is None
                else ProfileValue(
                    visibility.visibility,
                    visibility.source,
                    visibility.confidence,
                    visibility.confirmed,
                ),
            )
        )
    return tuple(result)


def _legacy_width_anchors(anchors: Sequence[ProfileAnchor]) -> tuple[WidthAnchor, ...]:
    return tuple(
        WidthAnchor(
            a.position,
            a.width.value,
            a.width.source,
            a.width.confidence,
            a.width.confirmed,
        )
        for a in anchors
        if a.width is not None
    )


def _legacy_visibility_anchors(
    anchors: Sequence[ProfileAnchor],
) -> tuple[VisibilityAnchor, ...]:
    return tuple(
        VisibilityAnchor(
            a.position,
            a.visibility.value,
            a.visibility.source,
            a.visibility.confidence,
            a.visibility.confirmed,
        )
        for a in anchors
        if a.visibility is not None
    )


def _validate_shared_anchors(
    anchors: Sequence[ProfileAnchor],
    *,
    min_width: float | None,
    max_width: float | None,
) -> None:
    _validate_positions(
        positions=[anchor.position for anchor in anchors], field="anchors"
    )
    for index, anchor in enumerate(anchors):
        if anchor.width is None and anchor.visibility is None:
            raise ValueError(f"anchors[{index}] must contain width or visibility")
        if anchor.width is not None:
            _validate_profile_value(anchor.width, field=f"anchors[{index}].width")
            if min_width is not None and anchor.width.value < min_width:
                raise ValueError("width anchor is below min_width")
            if max_width is not None and anchor.width.value > max_width:
                raise ValueError("width anchor exceeds max_width")
        if anchor.visibility is not None:
            _validate_profile_value(
                anchor.visibility, field=f"anchors[{index}].visibility"
            )


def _validate_profile_value(value: ProfileValue, *, field: str) -> None:
    if not math.isfinite(value.value):
        raise ValueError(f"{field}.value must be finite")
    _validate_confidence(value.confidence)
    _validate_source(value.source)
    if not isinstance(value.confirmed, bool):
        raise ValueError(f"{field}.confirmed must be bool")
    if field.endswith(".width") and value.value <= 0:
        raise ValueError(f"{field}.value must be positive")
    if field.endswith(".visibility") and not 0.0 <= value.value <= 1.0:
        raise ValueError(f"{field}.value must be within [0, 1]")


def _measurement_overrides_from_json(
    value: object,
) -> tuple[tuple[MeasurementOverrideKey, float], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("measurement_overrides must be an object or null")
    overrides: list[tuple[MeasurementOverrideKey, float]] = []
    for key, raw in value.items():
        if key not in _MEASUREMENT_OVERRIDE_KEYS:
            raise ValueError(f"unsupported measurement override: {key!r}")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"measurement override {key!r} must be a number")
        overrides.append((cast(MeasurementOverrideKey, key), float(raw)))
    result = tuple(sorted(overrides))
    _validate_measurement_overrides(result)
    return result


def _validate_measurement_overrides(
    overrides: Sequence[tuple[MeasurementOverrideKey, float]],
) -> None:
    values: dict[str, float] = {}
    for key, value in overrides:
        if key not in _MEASUREMENT_OVERRIDE_KEYS:
            raise ValueError(f"unsupported measurement override: {key!r}")
        if key in values:
            raise ValueError(f"duplicate measurement override: {key!r}")
        if not math.isfinite(value):
            raise ValueError(f"measurement override {key!r} must be finite")
        if key == "contrast_factor":
            if not 0.0 <= value <= 1.0:
                raise ValueError("contrast_factor override must be within [0, 1]")
        elif value <= 0.0:
            raise ValueError(f"measurement override {key!r} must be positive")
        values[key] = value
    if values.get("min_width", 0.0) > values.get("max_width", math.inf):
        raise ValueError("min_width override must not exceed max_width override")


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
                0.0 if segment_length == 0 else (distance - start) / segment_length
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
    """Sample left and right boundaries without creating annotation Shapes.

    Samples use the normal of the centerline segment containing their arc
    position. Centerline vertices are included in the sample positions and use
    the intersection of the two adjacent offset edges, so a sharp turn keeps
    its mitered envelope instead of drawing a diagonal bevel through the turn.
    The first and last samples use the end-segment normal, which gives the
    resulting polygon a butt cap.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    lengths = cumulative_lengths(points)
    total = lengths[-1]
    vertex_positions = (
        [
            length / total
            for index, length in enumerate(lengths[1:-1], start=1)
            if length > lengths[index - 1] and lengths[index + 1] > length
        ]
        if total > 0
        else []
    )
    sample_count = max(samples, len(vertex_positions) + 2)
    positions = [index / (sample_count - 1) for index in range(sample_count)]
    used_indices: set[int] = set()
    for vertex_position in vertex_positions:
        if any(
            math.isclose(position, vertex_position, abs_tol=1e-12)
            for position in positions
        ):
            continue
        candidates = (
            index for index in range(1, sample_count - 1) if index not in used_indices
        )
        index = min(
            candidates,
            key=lambda candidate: abs(positions[candidate] - vertex_position),
            default=None,
        )
        if index is not None:
            positions[index] = vertex_position
            used_indices.add(index)
    positions.sort()
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for position in positions:
        width = profile.evaluate_width(position)
        center = position_to_point(points, position)
        if width is None:
            left.append(center)
            right.append(center)
            continue
        corner = _profile_corner_boundary_points(
            points=points,
            lengths=lengths,
            position=position,
            half_width=width / 2.0,
        )
        if corner is not None:
            left.append(corner[0])
            right.append(corner[1])
            continue
        dx, dy = _segment_tangent_at_position(points, lengths, position * total)
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


def _profile_corner_boundary_points(
    *,
    points: Sequence[Sequence[float]],
    lengths: Sequence[float],
    position: float,
    half_width: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return both offset-edge intersections at an interior polyline vertex."""
    if half_width <= 0.0:
        return None
    distance = position * lengths[-1]
    vertex_index = next(
        (
            index
            for index in range(1, len(points) - 1)
            if math.isclose(distance, lengths[index], abs_tol=1e-9)
            and lengths[index] > lengths[index - 1]
            and lengths[index + 1] > lengths[index]
        ),
        None,
    )
    if vertex_index is None:
        return None
    center = (float(points[vertex_index][0]), float(points[vertex_index][1]))
    incoming = _unit_direction(points[vertex_index - 1], points[vertex_index])
    outgoing = _unit_direction(points[vertex_index], points[vertex_index + 1])
    if incoming is None or outgoing is None:
        return None

    result: list[tuple[float, float]] = []
    for side in (1.0, -1.0):
        incoming_normal = (-incoming[1] * side, incoming[0] * side)
        outgoing_normal = (-outgoing[1] * side, outgoing[0] * side)
        incoming_point = (
            center[0] + incoming_normal[0] * half_width,
            center[1] + incoming_normal[1] * half_width,
        )
        outgoing_point = (
            center[0] + outgoing_normal[0] * half_width,
            center[1] + outgoing_normal[1] * half_width,
        )
        denominator = _vector_cross(incoming, outgoing)
        if abs(denominator) < 1e-9:
            result.append(incoming_point)
            continue
        distance_along_incoming = (
            _vector_cross(
                (
                    outgoing_point[0] - incoming_point[0],
                    outgoing_point[1] - incoming_point[1],
                ),
                outgoing,
            )
            / denominator
        )
        intersection = (
            incoming_point[0] + incoming[0] * distance_along_incoming,
            incoming_point[1] + incoming[1] * distance_along_incoming,
        )
        # A nearly reversing or extremely acute turn can create an impractical
        # miter spike. Keep those cases bounded; ordinary right-angle turns
        # remain fully mitered and include the corner envelope.
        if math.dist(intersection, center) > 4.0 * half_width:
            result.append(incoming_point)
        else:
            result.append(intersection)
    return result[0], result[1]


def _unit_direction(
    start: Sequence[float], end: Sequence[float]
) -> tuple[float, float] | None:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None
    return dx / length, dy / length


def _vector_cross(left: Sequence[float], right: Sequence[float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def profile_boundary_polygon(
    profile: LineProfile,
    points: Sequence[Sequence[float]],
    *,
    samples: int = 64,
) -> tuple[tuple[float, float], ...]:
    """Return a closed-region outline with butt caps at both endpoints.

    The first boundary runs forward and the second runs backward. Consumers
    should close the path after drawing these vertices; the closing segments
    are the explicit perpendicular butt caps.
    """
    left, right = profile_boundary_points(profile, points, samples=samples)
    return (*left, *reversed(right))


def _segment_tangent_at_position(
    points: Sequence[Sequence[float]],
    lengths: Sequence[float],
    distance: float,
) -> tuple[float, float]:
    """Return a non-zero centerline segment tangent for an arc distance."""
    for index, (start, end) in enumerate(zip(lengths, lengths[1:])):
        if end <= start:
            continue
        if distance <= end or index == len(lengths) - 2:
            left, right = points[index], points[index + 1]
            return float(right[0] - left[0]), float(right[1] - left[1])
    return _fallback_tangent(points)


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
    remapped = [
        dataclasses.replace(
            anchor,
            position=point_to_position(
                new_points, position_to_point(old_points, anchor.position)
            ),
        )
        for anchor in profile.anchors
    ]
    return _replace_profile_anchors(profile, _deduplicate_shared_anchors(remapped))


def reverse_profile(profile: LineProfile) -> LineProfile:
    return _replace_profile_anchors(
        profile,
        [
            dataclasses.replace(anchor, position=1.0 - anchor.position)
            for anchor in reversed(profile.anchors)
        ],
    )


def crop_profile(
    profile: LineProfile, start_position: float, end_position: float
) -> LineProfile:
    """Keep an interval of a profile and renormalize it to ``[0, 1]``."""
    _validate_position(start_position, field="start_position")
    _validate_position(end_position, field="end_position")
    if start_position >= end_position:
        raise ValueError("start_position must be less than end_position")
    return _replace_profile_anchors(
        profile,
        _crop_shared_anchors(profile, start_position, end_position),
    )


def extend_profile(
    profile: LineProfile, start_position: float, end_position: float
) -> LineProfile:
    """Place an existing profile inside an extended normalized path interval."""
    _validate_position(start_position, field="start_position")
    _validate_position(end_position, field="end_position")
    if start_position >= end_position:
        raise ValueError("start_position must be less than end_position")
    scale = end_position - start_position
    return _replace_profile_anchors(
        profile,
        [
            dataclasses.replace(
                anchor, position=start_position + anchor.position * scale
            )
            for anchor in profile.anchors
        ],
    )


def resample_profile(profile: LineProfile, positions: Sequence[float]) -> LineProfile:
    """Sample both curves at fixed normalized positions without changing values."""
    normalized_positions = [float(position) for position in positions]
    _validate_positions(positions=normalized_positions, field="positions")
    anchors: list[ProfileAnchor] = []
    for position in normalized_positions:
        original = next(
            (
                anchor
                for anchor in profile.anchors
                if math.isclose(anchor.position, position, abs_tol=1e-9)
            ),
            None,
        )
        width = (
            original.width
            if original is not None
            else _auto_profile_value(profile.evaluate_width(position), "width")
        )
        visibility = (
            original.visibility
            if original is not None
            else _auto_profile_value(
                profile.evaluate_visibility(position), "visibility"
            )
        )
        if width is not None or visibility is not None:
            anchors.append(ProfileAnchor(position, width, visibility))
    return _replace_profile_anchors(profile, anchors)


def _auto_profile_value(value: float | None, property_name: str) -> ProfileValue | None:
    if value is None:
        return None
    return ProfileValue(value, "auto", 0.0, False)


def insert_width_anchor(
    profile: LineProfile,
    position: float,
    *,
    width: float | None = None,
    source: AnchorSource = "auto",
    confidence: float = 0.0,
    confirmed: bool = False,
) -> LineProfile:
    """Insert one width anchor, deriving its value from the current curve."""
    _validate_position(position, field="position")
    if any(
        math.isclose(anchor.position, position, abs_tol=1e-9)
        for anchor in profile.width_anchors
    ):
        raise ValueError("width anchor position already exists")
    value = profile.evaluate_width(position) if width is None else width
    if value is None:
        raise ValueError("cannot insert a width anchor into an empty curve")
    value_obj = ProfileValue(value, source, confidence, confirmed)
    shared = list(profile.anchors)
    existing = next(
        (a for a in shared if math.isclose(a.position, position, abs_tol=1e-9)), None
    )
    if existing is None:
        shared.append(ProfileAnchor(position, width=value_obj))
    else:
        shared[shared.index(existing)] = dataclasses.replace(existing, width=value_obj)
    return _replace_profile_anchors(profile, shared)


def insert_visibility_anchor(
    profile: LineProfile,
    position: float,
    *,
    visibility: float | None = None,
    source: AnchorSource = "auto",
    confidence: float = 0.0,
    confirmed: bool = False,
) -> LineProfile:
    """Insert one visibility anchor, deriving its value from the current curve."""
    _validate_position(position, field="position")
    if any(
        math.isclose(anchor.position, position, abs_tol=1e-9)
        for anchor in profile.visibility_anchors
    ):
        raise ValueError("visibility anchor position already exists")
    value = profile.evaluate_visibility(position) if visibility is None else visibility
    if value is None:
        raise ValueError("cannot insert a visibility anchor into an empty curve")
    value_obj = ProfileValue(value, source, confidence, confirmed)
    shared = list(profile.anchors)
    existing = next(
        (a for a in shared if math.isclose(a.position, position, abs_tol=1e-9)), None
    )
    if existing is None:
        shared.append(ProfileAnchor(position, visibility=value_obj))
    else:
        shared[shared.index(existing)] = dataclasses.replace(
            existing, visibility=value_obj
        )
    return _replace_profile_anchors(profile, shared)


def remove_width_anchor(profile: LineProfile, index: int) -> LineProfile:
    """Remove a width anchor while retaining the rest of the profile."""
    indices = [
        anchor_index
        for anchor_index, anchor in enumerate(profile.anchors)
        if anchor.width is not None
    ]
    if not 0 <= index < len(indices):
        raise IndexError("width anchor index is out of range")
    return remove_profile_property(profile, indices[index], "width")


def remove_visibility_anchor(profile: LineProfile, index: int) -> LineProfile:
    """Remove a visibility anchor while retaining the rest of the profile."""
    indices = [
        anchor_index
        for anchor_index, anchor in enumerate(profile.anchors)
        if anchor.visibility is not None
    ]
    if not 0 <= index < len(indices):
        raise IndexError("visibility anchor index is out of range")
    return remove_profile_property(profile, indices[index], "visibility")


def remove_profile_property(
    profile: LineProfile, index: int, property_name: Literal["width", "visibility"]
) -> LineProfile:
    """Remove one property while retaining the shared anchor when possible."""
    if property_name not in ("width", "visibility"):
        raise ValueError("property_name must be 'width' or 'visibility'")
    if not 0 <= index < len(profile.anchors):
        raise IndexError("profile anchor index is out of range")
    anchor = profile.anchors[index]
    updated = dataclasses.replace(anchor, **{property_name: None})
    anchors = list(profile.anchors)
    if updated.width is None and updated.visibility is None:
        del anchors[index]
    else:
        anchors[index] = updated
    return _replace_profile_anchors(profile, anchors)


def update_profile_anchor(
    profile: LineProfile,
    index: int,
    *,
    position: float | None = None,
    width: ProfileValue | None | object = ...,
    visibility: ProfileValue | None | object = ...,
) -> LineProfile:
    """Update a shared anchor without losing its other optional property."""
    if not 0 <= index < len(profile.anchors):
        raise IndexError("profile anchor index is out of range")
    anchor = profile.anchors[index]
    updated = dataclasses.replace(
        anchor,
        position=anchor.position if position is None else position,
        width=anchor.width if width is ... else width,
        visibility=anchor.visibility if visibility is ... else visibility,
    )
    anchors = list(profile.anchors)
    anchors[index] = updated
    return _replace_profile_anchors(profile, anchors)


def _replace_profile_anchors(
    profile: LineProfile, anchors: Sequence[ProfileAnchor]
) -> LineProfile:
    """Create a profile from one canonical, sorted shared-anchor sequence."""
    ordered = tuple(sorted(anchors, key=lambda anchor: anchor.position))
    return dataclasses.replace(
        profile,
        anchors=ordered,
    )


def _deduplicate_shared_anchors(
    anchors: Sequence[ProfileAnchor],
) -> tuple[ProfileAnchor, ...]:
    result: list[ProfileAnchor] = []
    for anchor in sorted(anchors, key=lambda item: item.position):
        if result and math.isclose(result[-1].position, anchor.position, abs_tol=1e-9):
            previous = result[-1]
            result[-1] = ProfileAnchor(
                position=previous.position,
                width=_prefer_profile_value(previous.width, anchor.width),
                visibility=_prefer_profile_value(
                    previous.visibility, anchor.visibility
                ),
            )
        else:
            result.append(anchor)
    return tuple(result)


def _prefer_profile_value(
    left: ProfileValue | None, right: ProfileValue | None
) -> ProfileValue | None:
    if left is None:
        return right
    if right is None:
        return left
    left_rank = (int(left.confirmed), int(left.source == "manual"), left.confidence)
    right_rank = (int(right.confirmed), int(right.source == "manual"), right.confidence)
    return right if right_rank > left_rank else left


def split_profile(
    profile: LineProfile, split_position: float
) -> tuple[LineProfile, LineProfile]:
    """Split a profile at a normalized position and normalize both halves."""
    _validate_position(split_position, field="split_position")
    if split_position <= 0.0 or split_position >= 1.0:
        raise ValueError("split_position must be strictly within (0, 1)")
    return (
        _replace_profile_anchors(
            profile, _split_shared_anchors(profile, split_position, left=True)
        ),
        _replace_profile_anchors(
            profile, _split_shared_anchors(profile, split_position, left=False)
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
    anchors = _merge_shared_anchor_sets(
        left.anchors,
        right.anchors,
        left_scale=left_scale,
        right_offset=right_offset,
        right_scale=right_scale,
        join_policy=join_policy,
    )
    return _replace_profile_anchors(
        left,
        anchors,
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
        anchors=tuple(
            dataclasses.replace(anchor, position=offset + anchor.position * scale)
            for anchor in profile.anchors
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


def _crop_anchor_set(
    anchors: tuple[Anchor, ...],
    evaluate: Callable[[float], float | None],
    start_position: float,
    end_position: float,
    make_anchor: type[WidthAnchor] | type[VisibilityAnchor],
) -> tuple[Anchor, ...]:
    if not anchors:
        return ()
    selected = [
        dataclasses.replace(
            anchor,
            position=(anchor.position - start_position)
            / (end_position - start_position),
        )
        for anchor in anchors
        if start_position <= anchor.position <= end_position
    ]
    for boundary, normalized in ((start_position, 0.0), (end_position, 1.0)):
        if any(math.isclose(anchor.position, normalized) for anchor in selected):
            continue
        value = evaluate(boundary)
        if value is None:
            continue
        if make_anchor is WidthAnchor:
            selected.append(
                WidthAnchor(
                    position=normalized,
                    width=value,
                    source="auto",
                    confidence=0.0,
                    confirmed=False,
                )
            )
        else:
            selected.append(
                VisibilityAnchor(
                    position=normalized,
                    visibility=value,
                    source="auto",
                    confidence=0.0,
                    confirmed=False,
                )
            )
    return _deduplicate_anchors(selected)


def _crop_shared_anchors(
    profile: LineProfile, start_position: float, end_position: float
) -> list[ProfileAnchor]:
    selected = [
        dataclasses.replace(
            anchor,
            position=(anchor.position - start_position)
            / (end_position - start_position),
        )
        for anchor in profile.anchors
        if start_position <= anchor.position <= end_position
    ]
    for boundary, normalized in ((start_position, 0.0), (end_position, 1.0)):
        if any(math.isclose(anchor.position, normalized) for anchor in selected):
            continue
        width = _auto_profile_value(profile.evaluate_width(boundary), "width")
        visibility = _auto_profile_value(
            profile.evaluate_visibility(boundary), "visibility"
        )
        if width is not None or visibility is not None:
            selected.append(ProfileAnchor(normalized, width, visibility))
    return list(_deduplicate_shared_anchors(selected))


def _resample_anchor_set(
    anchors: tuple[Anchor, ...],
    evaluate: Callable[[float], float | None],
    positions: list[float],
    make_anchor: type[WidthAnchor] | type[VisibilityAnchor],
) -> tuple[Anchor, ...]:
    result: list[Anchor] = []
    for position in positions:
        value = evaluate(position)
        if value is None:
            continue
        original = next(
            (
                anchor
                for anchor in anchors
                if math.isclose(anchor.position, position, abs_tol=1e-9)
            ),
            None,
        )
        if original is not None:
            result.append(dataclasses.replace(original, position=position))
        elif make_anchor is WidthAnchor:
            result.append(WidthAnchor(position, value, "auto", 0.0, False))
        else:
            result.append(VisibilityAnchor(position, value, "auto", 0.0, False))
    return tuple(result)


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


def _split_shared_anchors(
    profile: LineProfile, split_position: float, *, left: bool
) -> list[ProfileAnchor]:
    start, end = (0.0, split_position) if left else (split_position, 1.0)
    scale = split_position if left else 1.0 - split_position
    selected = [
        dataclasses.replace(
            anchor,
            position=(anchor.position if left else anchor.position - split_position)
            / scale,
        )
        for anchor in profile.anchors
        if start <= anchor.position <= end
    ]
    for boundary, normalized in ((start, 0.0), (end, 1.0)):
        if any(math.isclose(anchor.position, normalized) for anchor in selected):
            continue
        selected.append(
            ProfileAnchor(
                normalized,
                _auto_profile_value(profile.evaluate_width(boundary), "width"),
                _auto_profile_value(
                    profile.evaluate_visibility(boundary), "visibility"
                ),
            )
        )
    return list(_deduplicate_shared_anchors(selected))


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
        anchor for anchor in mapped_right if math.isclose(anchor.position, right_offset)
    ]
    if at_join_left and at_join_right:
        left_anchor, right_anchor = at_join_left[-1], at_join_right[0]
        if join_policy == "right":
            mapped_left.remove(left_anchor)
        elif join_policy == "average":
            value_name = "width" if hasattr(left_anchor, "width") else "visibility"
            average = (
                getattr(left_anchor, value_name) + getattr(right_anchor, value_name)
            ) / 2
            mapped_left[-1] = dataclasses.replace(
                left_anchor,
                **{
                    value_name: average,
                    "confidence": min(left_anchor.confidence, right_anchor.confidence),
                },
            )
            mapped_right.remove(right_anchor)
        else:
            mapped_right.remove(right_anchor)
    return _deduplicate_anchors(mapped_left + mapped_right)


def _merge_shared_anchor_sets(
    left: tuple[ProfileAnchor, ...],
    right: tuple[ProfileAnchor, ...],
    *,
    left_scale: float,
    right_offset: float,
    right_scale: float,
    join_policy: Literal["left", "right", "average"],
) -> tuple[ProfileAnchor, ...]:
    mapped = [
        dataclasses.replace(anchor, position=anchor.position * left_scale)
        for anchor in left
    ] + [
        dataclasses.replace(
            anchor, position=right_offset + anchor.position * right_scale
        )
        for anchor in right
    ]
    result: list[ProfileAnchor] = []
    for anchor in sorted(mapped, key=lambda item: item.position):
        if not result or not math.isclose(
            result[-1].position, anchor.position, abs_tol=1e-9
        ):
            result.append(anchor)
            continue
        previous = result[-1]
        if join_policy == "right":
            result[-1] = anchor
            continue
        if join_policy == "average":
            result[-1] = ProfileAnchor(
                previous.position,
                _average_profile_value(previous.width, anchor.width),
                _average_profile_value(previous.visibility, anchor.visibility),
            )
            continue
        result[-1] = ProfileAnchor(
            previous.position,
            _prefer_profile_value(previous.width, anchor.width),
            _prefer_profile_value(previous.visibility, anchor.visibility),
        )
    return tuple(result)


def _average_profile_value(
    left: ProfileValue | None, right: ProfileValue | None
) -> ProfileValue | None:
    if left is None:
        return right
    if right is None:
        return left
    return ProfileValue(
        (left.value + right.value) / 2.0,
        "manual" if left.source == right.source == "manual" else "auto",
        min(left.confidence, right.confidence),
        left.confirmed and right.confirmed,
    )


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
