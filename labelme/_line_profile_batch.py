from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from ._label_file import Annotation
from ._label_file import read_label_file
from ._label_file import write_label_file
from ._line_measurement import MEASUREMENT_VERSION
from ._line_measurement import MeasurementParameters
from ._line_measurement import MeasurementSample
from ._line_measurement import measure_line_profile
from ._line_profile import LineProfile
from ._line_profile import VisibilityAnchor
from ._line_profile import WidthAnchor
from ._utils.image import img_data_to_arr


@dataclasses.dataclass(frozen=True)
class BatchOptions:
    only_missing: bool = True
    only_unreviewed: bool = False
    dry_run: bool = False
    parameters: MeasurementParameters = dataclasses.field(
        default_factory=MeasurementParameters
    )
    cancel_check: Callable[[], bool] | None = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")


@dataclasses.dataclass(frozen=True)
class BatchItemResult:
    filename: str
    status: str
    processed_shapes: int = 0
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class BatchReport:
    items: tuple[BatchItemResult, ...]

    @property
    def failed(self) -> tuple[BatchItemResult, ...]:
        return tuple(item for item in self.items if item.status == "failed")

    @property
    def canceled(self) -> tuple[BatchItemResult, ...]:
        return tuple(item for item in self.items if item.status == "canceled")


class _BatchCanceled(Exception):
    pass


def measure_annotation_files(
    filenames: list[str],
    *,
    output_dir: str | None = None,
    options: BatchOptions | None = None,
) -> BatchReport:
    """Measure eligible files with per-file atomic writes and a report."""
    options = options or BatchOptions()
    results: list[BatchItemResult] = []
    for filename in filenames:
        if options.cancel_check is not None and options.cancel_check():
            results.append(BatchItemResult(filename, "canceled"))
            break
        for attempt in range(options.retry_count + 1):
            try:
                results.append(
                    _measure_annotation_file(
                        filename=filename, output_dir=output_dir, options=options
                    )
                )
                break
            except _BatchCanceled:
                results.append(BatchItemResult(filename, "canceled"))
                return BatchReport(tuple(results))
            except Exception as e:
                if attempt == options.retry_count:
                    results.append(
                        BatchItemResult(
                            filename,
                            "failed",
                            error=f"{type(e).__name__}: {e}",
                        )
                    )
    return BatchReport(tuple(results))


def _measure_annotation_file(
    *, filename: str, output_dir: str | None, options: BatchOptions
) -> BatchItemResult:
    annotation = read_label_file(filename=filename)
    image = img_data_to_arr(img_data=annotation.image_data)
    shapes = [dict(shape) for shape in annotation.shapes]
    processed = 0
    skipped_invalid = False
    for shape in shapes:
        if options.cancel_check is not None and options.cancel_check():
            raise _BatchCanceled
        if shape.get("shape_type") != "linestrip":
            continue
        if shape.get("line_profile_error") is not None:
            skipped_invalid = True
            continue
        profile = shape.get("line_profile")
        if options.only_missing and profile is not None:
            continue
        if (
            options.only_unreviewed
            and profile is not None
            and profile.reviewed
            and all(anchor.confidence >= 0.5 for anchor in profile.width_anchors)
        ):
            continue
        measurement = measure_line_profile(
            image, shape["points"], parameters=options.parameters
        )
        shape["line_profile"] = _profile_from_measurement(measurement.samples)
        processed += 1
    if processed and not options.dry_run:
        target = (
            Path(output_dir) / Path(filename).name
            if output_dir is not None
            else Path(filename)
        )
        if output_dir is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
        write_label_file(
            filename=str(target),
            annotation=Annotation(
                image_path=annotation.image_path,
                image_data=annotation.image_data,
                shapes=shapes,
                flags=annotation.flags,
                other_data=annotation.other_data,
            ),
            image_height=None,
            image_width=None,
            save_image_data=True,
        )
    status = "skipped-invalid" if skipped_invalid and not processed else "processed"
    if not processed and not skipped_invalid:
        status = "skipped"
    return BatchItemResult(filename, status, processed)


def _profile_from_measurement(
    samples: tuple[MeasurementSample, ...]
) -> LineProfile:
    return LineProfile(
        width_anchors=tuple(
            WidthAnchor(
                position=sample.position,
                width=sample.width,
                source="auto",
                confidence=sample.confidence,
                confirmed=False,
            )
            for sample in samples
        ),
        visibility_anchors=tuple(
            VisibilityAnchor(
                position=sample.position,
                visibility=sample.visibility,
                source="auto",
                confidence=sample.confidence,
                confirmed=False,
            )
            for sample in samples
        ),
        measurement_version=MEASUREMENT_VERSION,
    )
