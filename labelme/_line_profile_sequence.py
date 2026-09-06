from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Sequence

from ._shape import Shape


@dataclasses.dataclass(frozen=True)
class FrameShapeDifference:
    """Comparable, non-sensitive differences between two mapped Shapes."""

    index: int
    label: str | None
    centerline_max_displacement: float
    width_max_difference: float | None
    visibility_max_difference: float | None


def frame_sort_key(image_path: str) -> tuple[str, str, str]:
    """Return the stable ordering used for frame navigation."""
    basename = os.path.basename(image_path)
    return basename.casefold(), basename, image_path


def validate_frame_mapping(
    previous_shapes: Sequence[Shape], current_shapes: Sequence[Shape]
) -> None:
    """Reject unsafe positional frame mappings before copying profile data."""
    if len(previous_shapes) != len(current_shapes):
        raise ValueError("frame Shape counts do not match")
    for index, (previous, current) in enumerate(
        zip(previous_shapes, current_shapes)
    ):
        if previous.label != current.label:
            raise ValueError(f"frame Shape label mismatch at index {index}")
        if previous.shape_type != current.shape_type:
            raise ValueError(f"frame Shape type mismatch at index {index}")
        if len(previous.points) != len(current.points):
            raise ValueError(f"frame Shape path mismatch at index {index}")
        if previous.shape_type == "linestrip":
            previous_mode = (
                None
                if previous.line_profile is None
                else previous.line_profile.path_mode
            )
            current_mode = (
                None if current.line_profile is None else current.line_profile.path_mode
            )
            if (
                previous_mode is not None
                and current_mode is not None
                and previous_mode != current_mode
            ):
                raise ValueError(f"frame profile mode mismatch at index {index}")


def transfer_frame_profiles(
    previous_shapes: Sequence[Shape], current_shapes: Sequence[Shape]
) -> list[Shape]:
    """Copy only profiles into cloned current-frame Shapes after validation."""
    validate_frame_mapping(previous_shapes, current_shapes)
    result = [shape.copy() for shape in current_shapes]
    for previous, current in zip(previous_shapes, result):
        if current.shape_type == "linestrip":
            current.line_profile = previous.line_profile
    return result


def compare_frames(
    previous_shapes: Sequence[Shape], current_shapes: Sequence[Shape]
) -> tuple[FrameShapeDifference, ...]:
    """Return numeric differences for a user-facing transfer preview."""
    validate_frame_mapping(previous_shapes, current_shapes)
    differences: list[FrameShapeDifference] = []
    for index, (previous, current) in enumerate(
        zip(previous_shapes, current_shapes)
    ):
        displacement = 0.0
        if len(previous.points):
            displacement = float(
                max(
                    math.hypot(*(right - left))
                    for left, right in zip(previous.points, current.points)
                )
            )
        width_difference = _curve_difference(
            previous.line_profile,
            current.line_profile,
            "evaluate_width",
        )
        visibility_difference = _curve_difference(
            previous.line_profile,
            current.line_profile,
            "evaluate_visibility",
        )
        differences.append(
            FrameShapeDifference(
                index=index,
                label=current.label,
                centerline_max_displacement=displacement,
                width_max_difference=width_difference,
                visibility_max_difference=visibility_difference,
            )
        )
    return tuple(differences)


def _curve_difference(
    previous: object,
    current: object,
    method_name: str,
) -> float | None:
    if previous is None or current is None:
        return None
    positions = [index / 20 for index in range(21)]
    previous_evaluate = getattr(previous, method_name)
    current_evaluate = getattr(current, method_name)
    values = [
        abs(previous_evaluate(position) - current_evaluate(position))
        for position in positions
        if previous_evaluate(position) is not None
        and current_evaluate(position) is not None
    ]
    return max(values) if values else None
