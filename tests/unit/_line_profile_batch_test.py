from __future__ import annotations

from pathlib import Path

import pytest

import labelme._line_profile_batch as batch
from labelme._label_file import Annotation
from labelme._label_file import ShapeDict
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._line_profile_batch import BatchOptions
from labelme._line_profile_batch import measure_annotation_files


def test_batch_measurement_supports_dry_run_and_atomic_output(
    data_path: Path, tmp_path: Path
) -> None:
    source = read_label_file(str(data_path / "annotated/2011_000003.json"))
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
    assert read_label_file(str(input_file)).shapes[0].get("line_profile") is None

    output_dir = tmp_path / "out"
    report = measure_annotation_files(
        [str(input_file)], output_dir=str(output_dir)
    )

    assert report.items[0].processed_shapes == 1
    output = read_label_file(str(output_dir / input_file.name))
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
        return original(filename)

    monkeypatch.setattr(batch, "read_label_file", flaky_read_label_file)
    report = measure_annotation_files(
        [str(source)], options=BatchOptions(dry_run=True, retry_count=1)
    )

    assert calls == 2
    assert report.items[0].status == "skipped"
