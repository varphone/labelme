from __future__ import annotations

import dataclasses
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from typing import cast

import numpy as np
import PIL.Image

from ._label_file import Annotation
from ._label_file import read_label_file
from ._label_file import write_label_file
from ._shape_dict import ShapeDict

_ZERO_LENGTH_EPSILON: Final = np.finfo(np.float64).eps


@dataclass(frozen=True)
class AnnotationBatchReport:
    files_updated: int
    shapes_updated: int
    canceled: bool = False


class _AnnotationBatchCanceled(Exception):
    pass


def rename_label_in_files(
    *,
    filenames: list[str],
    old_label: str,
    new_label: str,
    save_image_data: bool,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AnnotationBatchReport:
    def transform(
        annotation: Annotation, check_canceled: Callable[[], bool]
    ) -> tuple[Annotation, int]:
        shapes: list[ShapeDict] = []
        changed = 0
        for shape in annotation.shapes:
            if check_canceled():
                raise _AnnotationBatchCanceled
            updated = dict(shape)
            if shape["label"] == old_label:
                updated["label"] = new_label
                changed += 1
            shapes.append(cast(ShapeDict, updated))
        return dataclasses.replace(annotation, shapes=shapes), changed

    return _run_annotation_batch(
        filenames=filenames,
        transform=transform,
        save_image_data=save_image_data,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def set_circle_radius_in_files(
    *,
    filenames: list[str],
    label: str,
    radius: float,
    save_image_data: bool,
    cancel_check: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AnnotationBatchReport:
    def transform(
        annotation: Annotation, check_canceled: Callable[[], bool]
    ) -> tuple[Annotation, int]:
        shapes: list[ShapeDict] = []
        changed = 0
        for shape in annotation.shapes:
            if check_canceled():
                raise _AnnotationBatchCanceled
            updated = dict(shape)
            if shape["label"] == label and shape["shape_type"] == "circle":
                updated["points"] = _circle_points_with_radius(
                    points=shape["points"], radius=radius
                )
                changed += 1
            shapes.append(cast(ShapeDict, updated))
        return dataclasses.replace(annotation, shapes=shapes), changed

    return _run_annotation_batch(
        filenames=filenames,
        transform=transform,
        save_image_data=save_image_data,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )


def _run_annotation_batch(
    *,
    filenames: list[str],
    transform: Callable[[Annotation, Callable[[], bool]], tuple[Annotation, int]],
    save_image_data: bool,
    cancel_check: Callable[[], bool] | None,
    progress_callback: Callable[[int, int], None] | None,
) -> AnnotationBatchReport:
    staged: list[tuple[str, Annotation, int, int, int]] = []
    total = len(filenames)
    for index, filename in enumerate(filenames):
        check_canceled = cancel_check or (lambda: False)
        if check_canceled():
            return AnnotationBatchReport(0, 0, canceled=True)
        try:
            annotation = read_label_file(filename=filename)
            updated_annotation, changed = transform(annotation, check_canceled)
            if changed:
                image_height, image_width = _image_dimensions(
                    image_data=annotation.image_data
                )
                staged.append(
                    (
                        filename,
                        updated_annotation,
                        image_height,
                        image_width,
                        changed,
                    )
                )
        except _AnnotationBatchCanceled:
            return AnnotationBatchReport(0, 0, canceled=True)
        except Exception as error:
            raise RuntimeError(f"{filename}: {error}") from error
        if progress_callback is not None:
            progress_callback(index + 1, total)

    if cancel_check is not None and cancel_check():
        return AnnotationBatchReport(0, 0, canceled=True)

    for filename, annotation, image_height, image_width, _ in staged:
        try:
            write_label_file(
                filename=filename,
                annotation=annotation,
                image_height=image_height,
                image_width=image_width,
                save_image_data=save_image_data,
            )
        except Exception as error:
            raise RuntimeError(f"{filename}: {error}") from error

    return AnnotationBatchReport(
        files_updated=len(staged),
        shapes_updated=sum(changed for *_, changed in staged),
    )


def _image_dimensions(*, image_data: bytes) -> tuple[int, int]:
    with PIL.Image.open(io.BytesIO(image_data)) as image:
        return image.height, image.width


def _circle_points_with_radius(
    *, points: np.ndarray | list[list[float]], radius: float
) -> list[list[float]]:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape != (2, 2):
        raise ValueError("circle shapes must have exactly two points")

    center_x = float(point_array[0, 0])
    center_y = float(point_array[0, 1])
    point_x = float(point_array[1, 0])
    point_y = float(point_array[1, 1])
    direction_x = point_x - center_x
    direction_y = point_y - center_y
    length = (direction_x**2 + direction_y**2) ** 0.5
    if length <= _ZERO_LENGTH_EPSILON:
        direction_x, direction_y = 1.0, 0.0
    else:
        direction_x /= length
        direction_y /= length
    return [
        [center_x, center_y],
        [center_x + direction_x * radius, center_y + direction_y * radius],
    ]
