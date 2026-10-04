from __future__ import annotations

from typing import NotRequired
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ._line_profile import LineProfile


class ShapeDict(TypedDict):
    label: str
    points: list[list[float]]
    shape_type: str
    flags: dict[str, bool]
    description: str
    group_id: int | None
    mask: NDArray[np.bool_] | None
    other_data: dict
    line_profile: NotRequired[LineProfile | None]
    line_profile_error: NotRequired[str | None]
