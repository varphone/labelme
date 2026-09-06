from __future__ import annotations

import copy
import dataclasses
import typing
from typing import Any
from typing import Final
from typing import Literal
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.interpolate
from loguru import logger

from .._line_profile import LineProfile

ShapeType: TypeAlias = Literal[
    "polygon",
    "rectangle",
    "oriented_rectangle",
    "point",
    "line",
    "circle",
    "linestrip",
    "points",
    "bezier2",
    "bezier3",
    "catmull_rom",
    "bspline",
    "mask",
]

# Shape types whose points form an open-or-closed polyline that a user can
# extend or shrink one vertex at a time.
POLYLINE_SHAPE_TYPES: Final[tuple[ShapeType, ...]] = ("polygon", "linestrip")
BEZIER_SHAPE_TYPES: Final[tuple[ShapeType, ...]] = ("bezier2", "bezier3")
SPLINE_SHAPE_TYPES: Final[tuple[ShapeType, ...]] = ("catmull_rom", "bspline")


def bezier_degree(shape_type: ShapeType) -> int:
    if shape_type == "bezier2":
        return 2
    if shape_type == "bezier3":
        return 3
    raise ValueError(f"Not a Bezier shape: {shape_type!r}")


def bezier_point(
    points: npt.NDArray[np.float64], t: float
) -> npt.NDArray[np.float64]:
    """Evaluate a quadratic or cubic Bezier curve at normalized position ``t``."""
    if len(points) not in (3, 4):
        raise ValueError(f"Bezier curves require 3 or 4 points, got {len(points)}")
    t = float(np.clip(t, 0.0, 1.0))
    u = 1.0 - t
    if len(points) == 3:
        return u * u * points[0] + 2.0 * u * t * points[1] + t * t * points[2]
    return (
        u**3 * points[0]
        + 3.0 * u**2 * t * points[1]
        + 3.0 * u * t**2 * points[2]
        + t**3 * points[3]
    )


def bezier_sample_points(
    points: npt.NDArray[np.float64], samples: int = 64
) -> npt.NDArray[np.float64]:
    if len(points) not in (3, 4):
        raise ValueError(f"Bezier curves require 3 or 4 points, got {len(points)}")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    return np.array([bezier_point(points, t) for t in np.linspace(0.0, 1.0, samples)])


def spline_sample_points(
    points: npt.NDArray[np.float64],
    shape_type: ShapeType,
    samples_per_segment: int = 24,
) -> npt.NDArray[np.float64]:
    """Sample an open multi-knot Catmull-Rom or cubic B-spline curve."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return points.copy()
    if shape_type not in SPLINE_SHAPE_TYPES:
        raise ValueError(f"Not a spline shape: {shape_type!r}")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2")

    sampled: list[npt.NDArray[np.float64]] = []
    if shape_type == "catmull_rom":
        padded = np.vstack((points[0], points, points[-1]))
        for i in range(len(points) - 1):
            p0, p1, p2, p3 = padded[i : i + 4]
            for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                t2, t3 = t * t, t * t * t
                sampled.append(
                    0.5
                    * (
                        (2 * p1)
                        + (-p0 + p2) * t
                        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                    )
                )
        sampled.append(points[-1])
        return np.asarray(sampled)

    degree = min(3, len(points) - 1)
    interior_count = len(points) - degree - 1
    knots = np.concatenate(
        (
            np.zeros(degree + 1),
            np.arange(1, interior_count + 1, dtype=np.float64),
            np.full(degree + 1, interior_count + 1, dtype=np.float64),
        )
    )
    curve = scipy.interpolate.BSpline(knots, points, degree)
    parameters = np.linspace(
        knots[degree],
        knots[-degree - 1],
        (len(points) - degree) * samples_per_segment + 1,
    )
    return np.asarray(curve(parameters), dtype=np.float64)


def line_profile_centerline(
    points: npt.ArrayLike, shape_type: ShapeType
) -> npt.NDArray[np.float64]:
    """Return the sampled polyline used for line-profile geometry."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if shape_type in BEZIER_SHAPE_TYPES:
        return bezier_sample_points(array, samples=128)
    if shape_type in SPLINE_SHAPE_TYPES:
        return spline_sample_points(array, shape_type, samples_per_segment=32)
    return array


# Point counts each shape type's finished geometry is defined by. A shape
# still being drawn holds fewer points than these until it is finalized.
CIRCLE_POINT_COUNT: Final = 2
LINE_POINT_COUNT: Final = 2
RECTANGLE_POINT_COUNT: Final = 2
ORIENTED_RECTANGLE_POINT_COUNT: Final = 4
MIN_LINESTRIP_POINT_COUNT: Final = 2
MIN_POLYGON_POINT_COUNT: Final = 3


@dataclasses.dataclass(eq=False)
class Shape:
    label: str | None = None
    group_id: int | None = None
    shape_type: ShapeType = "polygon"
    flags: dict[str, bool] | None = None
    description: str | None = None
    mask: npt.NDArray[np.bool_] | None = None
    points: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    point_labels: npt.NDArray[np.int_] = dataclasses.field(
        default_factory=lambda: np.empty((0,), dtype=np.int_)
    )
    other_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    line_profile: LineProfile | None = None
    line_profile_error: str | None = None
    closed: bool = False
    visible: bool = True

    def __post_init__(self) -> None:
        if self.shape_type not in typing.get_args(ShapeType):
            raise ValueError(f"Unexpected shape_type: {self.shape_type}")
        self.points = np.array(self.points, dtype=np.float64).reshape(-1, 2)
        self.point_labels = np.array(self.point_labels, dtype=np.int_).reshape(-1)
        if len(self.point_labels) == 0 and len(self.points) > 0:
            self.point_labels = np.ones(len(self.points), dtype=np.int_)

    def can_add_point(self) -> bool:
        return self.shape_type in POLYLINE_SHAPE_TYPES + SPLINE_SHAPE_TYPES

    def can_remove_point(self) -> bool:
        if not self.can_add_point():
            return False
        floor = {
            "polygon": MIN_POLYGON_POINT_COUNT,
            "linestrip": MIN_LINESTRIP_POINT_COUNT,
            "catmull_rom": 3,
            "bspline": 3,
        }[self.shape_type]
        return len(self.points) > floor

    def insert_point(self, *, i: int, point: npt.ArrayLike, label: int = 1) -> None:
        if not self.can_add_point():
            logger.warning(
                "Cannot add point to: shape_type={!r}, len(points)={:d}",
                self.shape_type,
                len(self.points),
            )
            return
        old_points = self.points.copy()
        new_points = np.insert(
            self.points, i, np.asarray(point, dtype=np.float64).reshape(2), axis=0
        )
        new_profile = self._line_profile_after_points(
            old_points=old_points, new_points=new_points
        )
        self.points = new_points
        self.line_profile = new_profile
        self.point_labels = np.insert(self.point_labels, i, label)

    def remove_point(self, *, i: int) -> None:
        if not self.can_remove_point():
            logger.warning(
                "Cannot remove point from: shape_type={!r}, len(points)={:d}",
                self.shape_type,
                len(self.points),
            )
            return
        if self.shape_type in SPLINE_SHAPE_TYPES and len(self.points) <= 3:
            return
        old_points = self.points.copy()
        new_points = np.delete(self.points, i, axis=0)
        new_profile = self._line_profile_after_points(
            old_points=old_points, new_points=new_points
        )
        self.points = new_points
        self.point_labels = np.delete(self.point_labels, i)
        self.line_profile = new_profile

    def move_vertex(self, *, i: int, pos: npt.ArrayLike) -> None:
        old_points = self.points.copy()
        new_points = self.points.copy()
        new_points[i] = np.asarray(pos, dtype=np.float64).reshape(2)
        new_profile = self._line_profile_after_points(
            old_points=old_points, new_points=new_points
        )
        self.points = new_points
        self.line_profile = new_profile

    def translate(self, *, offset: npt.ArrayLike) -> None:
        self.points = self.points + np.asarray(offset, dtype=np.float64).reshape(2)

    def _line_profile_after_points(
        self,
        *,
        old_points: npt.NDArray[np.float64],
        new_points: npt.NDArray[np.float64],
    ) -> LineProfile | None:
        if self.line_profile is None or self.shape_type not in (
            "line",
            "linestrip",
            "bezier2",
            "bezier3",
            "catmull_rom",
            "bspline",
        ):
            return self.line_profile
        # Resolve through the package export at call time. Besides preserving
        # the historical ``labelme._shape.remap_profile`` extension point,
        # this keeps profile-edit failures atomic for callers that replace it.
        from . import remap_profile as profile_remap

        return profile_remap(
            profile=self.line_profile,
            old_points=line_profile_centerline(old_points, self.shape_type),
            new_points=line_profile_centerline(new_points, self.shape_type),
        )

    def copy(self) -> Shape:
        return copy.deepcopy(self)
