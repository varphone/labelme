from __future__ import annotations

from pathlib import Path

import pytest

import labelme._line_profile_batch as batch
from labelme._label_file import Annotation
from labelme._label_file import ShapeDict
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._line_measurement import MeasurementParameters
from labelme._line_profile import LineProfile
from labelme._line_profile import ProfileAnchor
from labelme._line_profile import ProfileValue
from labelme._line_profile_batch import BatchOptions
from labelme._line_profile_batch import measure_annotation_files
from labelme._line_profile_batch import preview_annotation_files
from labelme._line_profile_batch import rollback_batch


def test_profile_override_can_disable_global_fixed_width() -> None:
    parameters = MeasurementParameters(fixed_width=12.0)
    profile = LineProfile(measurement_overrides=(("fixed_width", 0.0),))

    merged = batch._parameters_for_profile(parameters, profile)

    assert merged.fixed_width is None


def test_batch_measurement_supports_dry_run_and_atomic_output(
    data_path: Path, tmp_path: Path
) -> None:
    source = read_label_file(filename=str(data_path / "annotated/2011_000003.json"))
    input_file = tmp_path / "input.json"
    shape = ShapeDict(
        label="stripe",
        points=[[20.0, 20.0], [80.0, 20.0]],
        shape_type="linestrip",
        flags={},
        description="",
        group_id=None,
        mask=None,
        other_data={},
    )
    write_label_file(
        filename=str(input_file),
        annotation=Annotation(
            image_path=source.image_path,
            image_data=source.image_data,
            shapes=[shape],
            flags={},
            other_data={},
        ),
        image_height=None,
        image_width=None,
        save_image_data=True,
    )

    dry_run = measure_annotation_files(
        [str(input_file)], options=BatchOptions(dry_run=True)
    )
    assert dry_run.items[0].status == "processed"
    assert read_label_file(filename=str(input_file)).shapes[0].get("line_profile") is None

    output_dir = tmp_path / "out"
    report = measure_annotation_files(
        [str(input_file)], output_dir=str(output_dir)
    )

    assert report.items[0].processed_shapes == 1
    output = read_label_file(filename=str(output_dir / input_file.name))
    assert output.shapes[0].get("line_profile") is not None


def test_batch_measurement_can_cancel_before_writing(data_path: Path) -> None:
    source = data_path / "annotated/2011_000003.json"

    report = measure_annotation_files(
        [str(source)], options=BatchOptions(cancel_check=lambda: True)
    )

    assert report.items[0].status == "canceled"
    assert report.canceled == report.items


def test_batch_measurement_retries_transient_read_failure(
    data_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = data_path / "annotated/2011_000003.json"
    original = batch.read_label_file
    calls = 0

    def flaky_read_label_file(filename: str) -> Annotation:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        return original(filename=filename)

    monkeypatch.setattr(batch, "read_label_file", flaky_read_label_file)
    report = measure_annotation_files(
        [str(source)], options=BatchOptions(dry_run=True, retry_count=1)
    )

    assert calls == 2
    assert report.items[0].status == "skipped"


def test_batch_measurement_can_resume_from_a_completed_ordinal(
    data_path: Path,
) -> None:
    source = data_path / "annotated/2011_000003.json"

    report = measure_annotation_files(
        [str(source), str(source)],
        options=BatchOptions(dry_run=True, resume_from=1),
    )

    assert len(report.items) == 1
    assert report.items[0].filename == str(source)


def test_batch_preview_can_be_confirmed_without_writing_first(
    data_path: Path, tmp_path: Path
) -> None:
    source = data_path / "annotated/2011_000003.json"
    output_dir = tmp_path / "out"
    seen: list[batch.BatchReport] = []

    report = measure_annotation_files(
        [str(source)],
        output_dir=str(output_dir),
        options=BatchOptions(
            confirm_check=lambda preview: seen.append(preview) or True
        ),
    )

    assert len(seen) == 1
    assert seen[0].is_preview is True
    assert report.confirmed is True


def test_batch_preview_can_be_rejected_without_writing(
    data_path: Path, tmp_path: Path
) -> None:
    source = data_path / "annotated/2011_000003.json"
    output_dir = tmp_path / "out"

    preview = preview_annotation_files([str(source)])
    report = measure_annotation_files(
        [str(source)],
        output_dir=str(output_dir),
        options=BatchOptions(confirm_check=lambda _: False),
    )

    assert preview.items == report.items
    assert report.is_preview is True
    assert report.confirmed is False
    assert not output_dir.exists()


def test_batch_report_can_restore_an_overwritten_output(
    data_path: Path, tmp_path: Path
) -> None:
    source = read_label_file(filename=str(data_path / "annotated/2011_000003.json"))
    input_file = tmp_path / "input.json"
    write_label_file(
        filename=str(input_file),
        annotation=Annotation(
            image_path=source.image_path,
            image_data=source.image_data,
            shapes=[
                ShapeDict(
                    label="stripe",
                    points=[[20.0, 20.0], [80.0, 20.0]],
                    shape_type="linestrip",
                    flags={},
                    description="",
                    group_id=None,
                    mask=None,
                    other_data={},
                )
            ],
            flags={},
            other_data={},
        ),
        image_height=None,
        image_width=None,
        save_image_data=True,
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    output = output_dir / input_file.name
    sentinel = b"previous output"
    output.write_bytes(sentinel)

    report = measure_annotation_files(
        [str(input_file)],
        output_dir=str(output_dir),
        options=BatchOptions(
            only_missing=False,
            backup_dir=str(tmp_path / "backups"),
        ),
    )

    assert report.items[0].backup_filename is not None
    assert rollback_batch(report) == (str(output),)
    assert output.read_bytes() == sentinel


def test_fill_skips_existing_profiles_and_rebuild_preserves_manual_anchors(
    data_path: Path, tmp_path: Path
) -> None:
    source = read_label_file(filename=str(data_path / "annotated/2011_000003.json"))

    def write_input(path: Path, profile: LineProfile | None) -> None:
        shape = ShapeDict(
            label="stripe",
            points=[[20.0, 20.0], [80.0, 20.0]],
            shape_type="linestrip",
            flags={},
            description="",
            group_id=None,
            mask=None,
            other_data={},
        )
        if profile is not None:
            shape["line_profile"] = profile
        write_label_file(
            filename=str(path),
            annotation=Annotation(
                image_path=source.image_path,
                image_data=source.image_data,
                shapes=[shape],
                flags={},
                other_data={},
            ),
            image_height=None,
            image_width=None,
            save_image_data=True,
        )

    missing = tmp_path / "missing.json"
    existing = tmp_path / "existing.json"
    manual_profile = LineProfile(
        anchors=(ProfileAnchor(0.5, width=ProfileValue(12.0, "manual", 1.0, True)),)
    )
    write_input(missing, None)
    write_input(existing, manual_profile)

    fill = measure_annotation_files(
        [str(missing), str(existing)],
        options=BatchOptions(),
    )
    assert [item.status for item in fill.items] == ["processed", "skipped"]
    assert read_label_file(filename=str(missing)).shapes[0]["line_profile"] is not None
    assert read_label_file(filename=str(existing)).shapes[0]["line_profile"] == manual_profile

    rebuild = measure_annotation_files(
        [str(existing)],
        options=BatchOptions(only_missing=False),
    )
    assert rebuild.items[0].status == "processed"
    rebuilt_profile = read_label_file(filename=str(existing)).shapes[0]["line_profile"]
    assert rebuilt_profile is not None
    assert any(
        anchor.width == manual_profile.anchors[0].width
        for anchor in rebuilt_profile.anchors
        if anchor.width is not None
    )
    assert sum(anchor.width is not None for anchor in rebuilt_profile.anchors) > 1
