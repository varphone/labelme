from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtCore import QSize
from pytestqt.qtbot import QtBot

from labelme import _app
from labelme._label_file import Annotation
from labelme._label_file import ShapeDict
from labelme._label_file import read_label_file
from labelme._label_file import write_label_file
from labelme._line_profile import LineProfile
from labelme._line_profile import WidthAnchor
from labelme._shape import Shape

from .conftest import MainWinFactory
from .conftest import show_window_and_wait_for_imagedata


@pytest.mark.gui
def test_profile_preview_zoom_and_undo_preserve_centerline(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
) -> None:
    win = main_win(
        file_or_dir=str(data_path / "raw" / "2011_000003.jpg"),
        size=QSize(900, 700),
    )
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    shape = Shape(
        label="stripe",
        shape_type="linestrip",
        points=np.array([[10.0, 20.0], [180.0, 20.0]], dtype=np.float64),
        line_profile=LineProfile(
            width_anchors=(WidthAnchor(0.0, 12.0, "manual", 1.0, True),)
        ),
    )
    win._load_shapes([shape])
    win._canvas_widgets.canvas.select_shapes([shape])

    original_points = shape.points.copy()
    for zoom in (25.0, 100.0, 400.0):
        win._set_zoom(zoom)
        assert (
            win._canvas_widgets.canvas._find_line_profile_anchor_at_point(
                QPointF(10.0, 20.0)
            )
            is not None
        )

    win.clear_selected_line_profile()
    assert shape.line_profile is None
    win.undo_shape_edit()
    restored = win._canvas_widgets.canvas.shapes[0]
    assert restored.line_profile is not None
    np.testing.assert_array_equal(restored.points, original_points)
    assert win._docks.label_list[0].shape().line_profile is not None


@pytest.mark.gui
def test_continuous_frame_profile_copy_requires_mapping_and_does_not_write_previous(
    main_win: MainWinFactory,
    qtbot: QtBot,
    data_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = read_label_file(str(data_path / "annotated" / "2011_000003.json"))
    image_bytes = source.image_data
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    source_shape: ShapeDict = {
        "label": "stripe",
        "points": [[10.0, 20.0], [180.0, 20.0]],
        "shape_type": "linestrip",
        "flags": {},
        "description": "",
        "group_id": None,
        "mask": None,
        "other_data": {},
        "line_profile": LineProfile(
            width_anchors=(WidthAnchor(0.0, 12.0, "manual", 1.0, True),)
        ),
    }
    target_shape: ShapeDict = {**source_shape}
    target_shape.pop("line_profile")
    for name, shapes in (("frame-01", [source_shape]), ("frame-02", [target_shape])):
        image_path = frames_dir / f"{name}.jpg"
        image_path.write_bytes(image_bytes)
        write_label_file(
            filename=str(frames_dir / f"{name}.json"),
            annotation=Annotation(
                image_path=image_path.name,
                image_data=image_bytes,
                shapes=shapes,
                flags={},
                other_data={},
            ),
            image_height=None,
            image_width=None,
            save_image_data=True,
        )
    previous_frame_bytes = (frames_dir / "frame-01.json").read_bytes()

    win = main_win(file_or_dir=frames_dir)
    show_window_and_wait_for_imagedata(qtbot=qtbot, win=win)
    assert win._docks.file_list.count() == 2
    win._docks.file_list.setCurrentRow(1)
    qtbot.waitUntil(
        lambda: win._image_path is not None and "frame-02" in win._image_path
    )

    monkeypatch.setattr(
        _app.QMessageBox,
        "question",
        lambda *args, **kwargs: _app.QMessageBox.StandardButton.Yes,
    )
    win.copy_profiles_from_previous_frame()

    current_shape = win._canvas_widgets.canvas.shapes[0]
    assert current_shape.line_profile is not None
    assert (frames_dir / "frame-01.json").read_bytes() == previous_frame_bytes
