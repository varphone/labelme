from __future__ import annotations

import json
from pathlib import Path

import PIL.Image
from pytestqt.qtbot import QtBot

from labelme._annotation_batch import rename_label_in_files
from labelme._widgets.annotation_batch_worker import AnnotationBatchWorker


def _write_annotation(
    *, directory: Path, name: str, shapes: list[dict[str, object]]
) -> Path:
    image_name = f"{name}.png"
    PIL.Image.new("RGB", (16, 12)).save(directory / image_name)
    label_path = directory / f"{name}.json"
    label_path.write_text(
        json.dumps(
            {
                "version": "4.0.0",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
            }
        )
    )
    return label_path


def test_rename_label_reports_file_progress_without_widgets(*, tmp_path: Path) -> None:
    first = _write_annotation(
        directory=tmp_path,
        name="first",
        shapes=[
            {"label": "old", "shape_type": "point", "points": [[1, 2]]},
        ],
    )
    second = _write_annotation(
        directory=tmp_path,
        name="second",
        shapes=[
            {"label": "other", "shape_type": "point", "points": [[3, 4]]},
        ],
    )
    progress: list[tuple[int, int]] = []

    report = rename_label_in_files(
        filenames=[str(first), str(second)],
        old_label="old",
        new_label="new",
        save_image_data=False,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert report.files_updated == 1
    assert report.shapes_updated == 1
    assert progress == [(1, 2), (2, 2)]
    assert json.loads(first.read_text())["shapes"][0]["label"] == "new"
    assert json.loads(second.read_text())["shapes"][0]["label"] == "other"


def test_worker_cancellation_does_not_write_staged_files(
    *, tmp_path: Path, qtbot: QtBot
) -> None:
    first = _write_annotation(
        directory=tmp_path,
        name="first",
        shapes=[
            {"label": "old", "shape_type": "point", "points": [[1, 2]]},
        ],
    )
    second = _write_annotation(
        directory=tmp_path,
        name="second",
        shapes=[
            {"label": "old", "shape_type": "point", "points": [[3, 4]]},
        ],
    )
    worker = AnnotationBatchWorker()
    canceled: list[bool] = []
    succeeded: list[object] = []
    worker.canceled.connect(lambda: canceled.append(True))
    worker.succeeded.connect(succeeded.append)
    worker.progress.connect(lambda _completed, _total: worker.cancel())

    worker.rename_label(
        filenames=[str(first), str(second)],
        old_label="old",
        new_label="new",
        save_image_data=False,
    )
    qtbot.wait(10)

    assert canceled == [True]
    assert succeeded == []
    assert json.loads(first.read_text())["shapes"][0]["label"] == "old"
    assert json.loads(second.read_text())["shapes"][0]["label"] == "old"
