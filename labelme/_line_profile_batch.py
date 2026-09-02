from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
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
from ._line_profile import ProfileAnchor
from ._line_profile import ProfileValue
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
    # A caller can show the dry-run report and decide whether writes may start.
    confirm_check: Callable[[BatchReport], bool] | None = None
    backup_dir: str | None = None
    progress_callback: Callable[[int, int], None] | None = None
    # The caller persists the last completed ordinal and passes it back after
    # cancellation; this keeps restart policy outside the file writer.
    resume_from: int = 0

    def __post_init__(self) -> None:
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.resume_from < 0:
            raise ValueError("resume_from must be non-negative")


@dataclasses.dataclass(frozen=True)
class BatchItemResult:
    filename: str
    status: str
    processed_shapes: int = 0
    error: str | None = None
    output_filename: str | None = None
    backup_filename: str | None = None


@dataclasses.dataclass(frozen=True)
class BatchReport:
    items: tuple[BatchItemResult, ...]
    is_preview: bool = False
    confirmed: bool = True

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
    if options.confirm_check is not None and not options.dry_run:
        preview = preview_annotation_files(filenames, options=options)
        if not options.confirm_check(preview):
            return BatchReport(
                preview.items,
                is_preview=True,
                confirmed=False,
            )
    results: list[BatchItemResult] = []
    for index, filename in enumerate(filenames):
        if index < options.resume_from:
            continue
        if options.cancel_check is not None and options.cancel_check():
            result = BatchItemResult(filename, "canceled")
            results.append(result)
            _report_batch_progress(options, index + 1, len(filenames))
            break
        for attempt in range(options.retry_count + 1):
            try:
                result = _measure_annotation_file(
                    filename=filename, output_dir=output_dir, options=options
                )
                results.append(result)
                _report_batch_progress(options, index + 1, len(filenames))
                break
            except _BatchCanceled:
                result = BatchItemResult(filename, "canceled")
                results.append(result)
                _report_batch_progress(options, index + 1, len(filenames))
                return BatchReport(tuple(results))
            except Exception as e:
                if attempt == options.retry_count:
                    result = BatchItemResult(
                        filename,
                        "failed",
                        error=f"{type(e).__name__}: {e}",
                    )
                    results.append(result)
                    _report_batch_progress(options, index + 1, len(filenames))
    return BatchReport(tuple(results), is_preview=options.dry_run)


def preview_annotation_files(
    filenames: list[str], *, options: BatchOptions | None = None
) -> BatchReport:
    """Build a write-free report suitable for confirmation in a batch UI."""
    source = options or BatchOptions()
    return measure_annotation_files(
        filenames,
        options=dataclasses.replace(
            source,
            dry_run=True,
            confirm_check=None,
        ),
    )


def _measure_annotation_file(
    *, filename: str, output_dir: str | None, options: BatchOptions
) -> BatchItemResult:
    annotation = read_label_file(filename=filename)
    image = img_data_to_arr(img_data=annotation.image_data)
    shapes = [dict(shape) for shape in annotation.shapes]
    processed = 0
    skipped_invalid = False
    target: Path | None = None
    backup_filename: str | None = None
    for shape in shapes:
        if options.cancel_check is not None and options.cancel_check():
            raise _BatchCanceled
        if shape.get("shape_type") != "linestrip":
            continue
        if shape.get("line_profile_error") is not None:
            skipped_invalid = True
            continue
        profile = shape.get("line_profile")
        if options.only_missing and _profile_has_anchors(profile):
            continue
        if (
            options.only_unreviewed
            and profile is not None
            and profile.reviewed
            and all(
                value is None or value.confidence >= 0.5
                for anchor in profile.anchors
                for value in (anchor.width, anchor.visibility)
            )
        ):
            continue
        measurement = measure_line_profile(
            image,
            shape["points"],
            parameters=_parameters_for_profile(options.parameters, profile),
        )
        shape["line_profile"] = _profile_from_measurement(
            measurement.samples,
            existing=profile,
        )
        processed += 1
    if processed and not options.dry_run:
        target = (
            Path(output_dir) / Path(filename).name
            if output_dir is not None
            else Path(filename)
        )
        if output_dir is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
        if options.backup_dir is not None and target.exists():
            backup_root = Path(options.backup_dir)
            backup_root.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(str(target).encode()).hexdigest()[:12]
            backup_path = backup_root / f"{digest}-{target.name}"
            shutil.copy2(target, backup_path)
            backup_filename = str(backup_path)
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
    return BatchItemResult(
        filename,
        status,
        processed,
        output_filename=(
            None
            if not processed or options.dry_run or target is None
            else str(target)
        ),
        backup_filename=(None if options.dry_run else backup_filename),
    )


def rollback_batch(report: BatchReport) -> tuple[str, ...]:
    """Restore files whose pre-write backups are present in a batch report."""
    restored: list[str] = []
    for item in report.items:
        if item.output_filename is None or item.backup_filename is None:
            continue
        backup = Path(item.backup_filename)
        target = Path(item.output_filename)
        if not backup.is_file():
            continue
        temporary = Path(f"{target}.rollback.tmp")
        try:
            shutil.copy2(backup, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        restored.append(str(target))
    return tuple(restored)


def _profile_from_measurement(
    samples: tuple[MeasurementSample, ...],
    *,
    existing: LineProfile | None = None,
) -> LineProfile:
    manual = {
        anchor.position: anchor
        for anchor in (() if existing is None else existing.anchors)
    }
    measured: list[ProfileAnchor] = []
    for sample in samples:
        current = next(
            (anchor for position, anchor in manual.items() if abs(position - sample.position) <= 1e-6),
            None,
        )
        width = None if current is not None and current.width is not None and current.width.confirmed else ProfileValue(sample.width, "auto", sample.confidence, False)
        visibility = None if current is not None and current.visibility is not None and current.visibility.confirmed else ProfileValue(sample.visibility, "auto", sample.confidence, False)
        measured.append(ProfileAnchor(sample.position, width, visibility))
    merged: list[ProfileAnchor] = []
    for anchor in (*manual.values(), *measured):
        existing_at_position = next(
            (item for item in merged if abs(item.position - anchor.position) <= 1e-6),
            None,
        )
        if existing_at_position is None:
            merged.append(anchor)
            continue
        width = existing_at_position.width or anchor.width
        visibility = existing_at_position.visibility or anchor.visibility
        merged[merged.index(existing_at_position)] = ProfileAnchor(
            existing_at_position.position, width, visibility
        )
    return LineProfile(
        anchors=tuple(sorted(merged, key=lambda anchor: anchor.position)),
        min_width=None if existing is None else existing.min_width,
        max_width=None if existing is None else existing.max_width,
        measurement_version=MEASUREMENT_VERSION,
        measurement_overrides=()
        if existing is None
        else existing.measurement_overrides,
    )


def _profile_has_anchors(profile: object) -> bool:
    return isinstance(profile, LineProfile) and bool(
        profile.anchors
    )


def _parameters_for_profile(
    parameters: MeasurementParameters, profile: object
) -> MeasurementParameters:
    if not isinstance(profile, LineProfile) or not profile.measurement_overrides:
        return parameters
    return dataclasses.replace(parameters, **dict(profile.measurement_overrides))


def _report_batch_progress(options: BatchOptions, completed: int, total: int) -> None:
    if options.progress_callback is not None:
        options.progress_callback(completed, total)
